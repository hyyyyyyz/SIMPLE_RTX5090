#!/usr/bin/env bash
# Run the SIMPLE eval CLI over every downloaded simple-eval task, dr-levels 0-2.
#
# Usage:
#   scripts/eval-simple-batch.sh                      # all tasks, dr-levels 0 1 2
#   TASKS="G1WholebodyXMovePickTeleop-v0" scripts/eval-simple-batch.sh
#   LEVELS="0 2" scripts/eval-simple-batch.sh
#   LEVEL_PREFIX=level- scripts/eval-simple-batch.sh  # if splits are named level-N
#   PORT=22086 NUM_WORKERS=4 scripts/eval-simple-batch.sh
#   SKIP_TASKS="" scripts/eval-simple-batch.sh       # include the excluded tasks
#   FORCE=1 scripts/eval-simple-batch.sh              # rerun already-completed evals
#   DRY_RUN=1 scripts/eval-simple-batch.sh            # resolve + print, launch nothing
#
# Dataset dir names do not always equal the registered gym env id (see the
# resolution step below); the id actually used is printed as "env=" on each run.
#
# Runs that exit 0 are recorded under $MARKER_DIR and skipped on the next
# invocation, so an interrupted batch can simply be restarted.
#
# Entry point is chosen per task (see README "Run the SIMPLE Simulation Client"):
#   MP suffix   -> src/simple/cli/eval.py                with agent psi0
#   otherwise   -> src/simple/cli/eval_decoupled_wbc.py  with agent psi0_decoupled_wbc
#
# MAX_EPISODE_STEPS is unset by default (no cap -- the CLI's own None default):
#   MAX_EPISODE_STEPS=1000 scripts/eval-simple-batch.sh
#
# Extra args after -- are forwarded to the eval CLI:
#   scripts/eval-simple-batch.sh -- --max-episode-steps=500

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Isaac Sim (via GLFW) puts the controlling terminal into a non-canonical,
# echo-off mode and the rich Live display hides the cursor; a worker tearing down
# can do either after the eval process has already cleaned up. Snapshot the line
# settings now and make the shell the final authority, so an interrupted or
# crashed run never leaves the terminal cursorless or unable to echo typing.
SAVED_TTY_STATE=""
if [[ -t 0 ]]; then
    SAVED_TTY_STATE="$(stty -g 2>/dev/null || true)"
fi
restore_terminal() {
    local rc=$?
    if [[ -t 1 ]]; then
        tput cnorm 2>/dev/null || printf '\033[?25h'
    fi
    if [[ -n "$SAVED_TTY_STATE" ]]; then
        stty "$SAVED_TTY_STATE" 2>/dev/null || stty sane 2>/dev/null || true
    fi
    return $rc
}
trap restore_terminal EXIT INT TERM

PYTHON=${PYTHON:-.venv/bin/python}
DATA_ROOT=${DATA_ROOT:-data/evals/simple-eval}
# Per README: MP (motion-planning) tasks use standard eval, the rest decoupled WBC.
TELEOP_CLI=${TELEOP_CLI:-src/simple/cli/eval_decoupled_wbc.py}
TELEOP_POLICY=${TELEOP_POLICY:-psi0_decoupled_wbc}
MP_CLI=${MP_CLI:-src/simple/cli/eval.py}
MP_POLICY=${MP_POLICY:-psi0}
HOST=${HOST:-localhost}
PORT=${PORT:-22085}
SIM_MODE=${SIM_MODE:-mujoco_isaac}
DATA_FORMAT=${DATA_FORMAT:-lerobot}
# Unset by default: the flag is then omitted entirely so the eval CLI applies its
# own default of None (no cap). Set to an integer to impose a per-episode limit.
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-}
NUM_WORKERS=${NUM_WORKERS:-2}
LEVELS=${LEVELS:-"0 1 2"}
# Split dirs ship as dr-level-N (domain randomization levels).
LEVEL_PREFIX=${LEVEL_PREFIX:-dr-level-}
# Completed (task, split) pairs are marked here so reruns resume where they left off.
MARKER_DIR=${MARKER_DIR:-data/evals/.batch-done}

EXTRA_ARGS=()
if [[ "${1:-}" == "--" ]]; then
    shift
    EXTRA_ARGS=("$@")
fi

if [[ -n "${TASKS:-}" ]]; then
    read -r -a task_list <<<"$TASKS"
else
    mapfile -t task_list < <(cd "$DATA_ROOT" && ls -d */ 2>/dev/null | sed 's#/$##' | sort)
fi

if [[ ${#task_list[@]} -eq 0 ]]; then
    echo "No task directories found under $DATA_ROOT" >&2
    exit 1
fi

# Dataset directories are named after the task as of capture time, but several
# envs have since been renamed -- the CamelCase connectors "And"/"N" were dropped
# from the dataset names (XMoveAndPick -> XMovePick, PickNPlace -> PickPlace).
# Passing "simple/<dataset-dir>" straight to gym.make therefore dies with
# gymnasium.error.NameNotFound, and only *after* Isaac Sim has finished booting.
# Resolve every directory to a real env id here and fail fast if any cannot be.
declare -A TASK_ENV=()
resolved=$("$PYTHON" - "${task_list[@]}" <<'PY'
import difflib
import re
import sys

import gymnasium as gym

import simple.envs  # noqa: F401  -- registering simple/* is the import's purpose

# Explicit dataset-dir -> env-id overrides. Each was confirmed by comparing the
# dataset's meta/tasks.jsonl instruction against the env's task `description`.
ALIASES = {
    "G1WholebodyBendPickPlaceMP-v0": "G1WholebodyBendPickAndPlaceMP-v0",
    "G1WholebodyBendPickPlaceOnSofaMP-v0": "G1WholebodyBendPickAndPlaceOnSofaMP-v0",
    "G1WholebodyPickBendPlaceMP-v0": "G1WholebodyPickAndBendPlaceMP-v0",
    "G1WholebodyPickPlaceMP-v0": "G1WholebodyPickNPlaceMP-v0",
    "G1WholebodyXMoveHandoverMP-v0": "G1WholebodyXMoveAndHandoverMP-v0",
    "G1WholebodyXMovePickMP-v0": "G1WholebodyXMoveAndPickMP-v0",
}

known = {k.split("/", 1)[1] for k in gym.registry if k.startswith("simple/")}

# Same rename rule as above, applied generically so future datasets that lag a
# rename resolve on their own. Only ever used when it lands on exactly one env.
def drop_connectors(name: str) -> str:
    return re.sub(r"(?<=[a-z])(?:And|N)(?=[A-Z])", "", name)

by_norm: dict[str, list[str]] = {}
for env in known:
    by_norm.setdefault(drop_connectors(env), []).append(env)

unresolved = []
for task in sys.argv[1:]:
    if task in known:
        env = task
    elif task in ALIASES and ALIASES[task] in known:
        env = ALIASES[task]
    else:
        candidates = by_norm.get(drop_connectors(task), [])
        if len(candidates) != 1:
            unresolved.append((task, candidates))
            continue
        env = candidates[0]
    print(f"{task}\t{env}")

if unresolved:
    print("", file=sys.stderr)
    for task, candidates in unresolved:
        hint = candidates or difflib.get_close_matches(task, sorted(known), n=3, cutoff=0.6)
        detail = f"ambiguous, matches {candidates}" if candidates else f"did you mean: {', '.join(hint) or '(no close match)'}"
        print(f"error: no registered env for dataset dir '{task}' -- {detail}", file=sys.stderr)
    print(
        "\nAdd the correct dataset-dir -> env-id pair to ALIASES in "
        f"{sys.argv[0] or 'scripts/eval-simple-batch.sh'}, or drop the directory from $DATA_ROOT.",
        file=sys.stderr,
    )
    sys.exit(1)
PY
) || exit 1

while IFS=$'\t' read -r task env; do
    [[ -n "$task" ]] && TASK_ENV["$task"]="$env"
done <<<"$resolved"

for task in "${task_list[@]}"; do
    if [[ "${TASK_ENV[$task]}" != "$task" ]]; then
        echo "[MAP ] $task -> simple/${TASK_ENV[$task]}"
    fi
done

# Tasks excluded from the batch. Set SKIP_TASKS="" to run everything.
SKIP_TASKS=${SKIP_TASKS-"
"}
# shellcheck disable=SC2206  # intentional word splitting on whitespace/newlines
skip_list=($SKIP_TASKS)

declare -a ok_runs=() failed_runs=() skipped_runs=()

for task in "${task_list[@]}"; do
    excluded=0
    for s in ${skip_list[@]+"${skip_list[@]}"}; do
        [[ "$task" == "$s" ]] && excluded=1 && break
    done
    if [[ $excluded -eq 1 ]]; then
        echo "[SKIP] $task (excluded)"
        skipped_runs+=("$task (excluded)")
        continue
    fi

    for lvl in $LEVELS; do
        split="${LEVEL_PREFIX}${lvl}"
        data_dir="$DATA_ROOT/$task/$split"
        if [[ ! -d "$data_dir" ]]; then
            echo "[SKIP] $task $split (no data dir)"
            skipped_runs+=("$task/$split (no data dir)")
            continue
        fi

        # Motion-planning tasks (MP suffix, before any version tag) run the
        # standard eval entry with the plain psi0 agent; the rest run decoupled WBC.
        if [[ "$task" == *MP || "$task" == *MP-* ]]; then
            eval_cli="$MP_CLI"
            policy="$MP_POLICY"
        else
            eval_cli="$TELEOP_CLI"
            policy="$TELEOP_POLICY"
        fi

        # Resume support: a marker is written only after a run exits 0.
        marker="$MARKER_DIR/$task.$split.done"
        if [[ -z "${FORCE:-}" && -f "$marker" ]]; then
            echo "[SKIP] $task $split (already completed)"
            skipped_runs+=("$task/$split (already completed)")
            continue
        fi

        env_id="simple/${TASK_ENV[$task]}"

        # Only cap episode length when asked; otherwise leave the CLI's None default
        # in place (passing "--max-episode-steps=None" would fail its int parsing).
        step_args=()
        if [[ -n "$MAX_EPISODE_STEPS" && "$MAX_EPISODE_STEPS" != "None" ]]; then
            step_args=("--max-episode-steps=$MAX_EPISODE_STEPS")
        fi

        echo
        echo "=============================================================="
        echo "[RUN ] $task  $split  ($(basename "$eval_cli" .py) / $policy)  env=$env_id"
        echo "=============================================================="
        if [[ -n "${DRY_RUN:-}" ]]; then
            echo "[DRY ] $PYTHON $eval_cli $env_id $policy $split --data-dir=$data_dir" \
                 "${step_args[@]+"${step_args[@]}"}" "--num-workers=$NUM_WORKERS ..."
            skipped_runs+=("$task/$split (dry run)")
            continue
        fi
        "$PYTHON" "$eval_cli" \
            "$env_id" \
            "$policy" \
            "$split" \
            "--host=$HOST" \
            "--port=$PORT" \
            "--sim-mode=$SIM_MODE" \
            "--data-format=$DATA_FORMAT" \
            "--data-dir=$data_dir" \
            ${step_args[@]+"${step_args[@]}"} \
            "--num-workers=$NUM_WORKERS" \
            "${EXTRA_ARGS[@]}"
        rc=$?
        if [[ $rc -eq 0 ]]; then
            echo "[ OK ] $task $split"
            mkdir -p "$MARKER_DIR" && : >"$marker"
            ok_runs+=("$task/$split")
        else
            echo "[FAIL] $task $split (exit $rc)"
            failed_runs+=("$task/$split (exit $rc)")
        fi
    done
done

echo
echo "=============================== SUMMARY ==============================="
echo "ok:      ${#ok_runs[@]}"
echo "failed:  ${#failed_runs[@]}"
echo "skipped: ${#skipped_runs[@]}"
for r in "${failed_runs[@]}"; do echo "  FAIL $r"; done
for r in "${skipped_runs[@]}"; do echo "  SKIP $r"; done

[[ ${#failed_runs[@]} -eq 0 ]]
