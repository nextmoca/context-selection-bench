#!/usr/bin/env bash
# Reproduce the published GPT-4o AgentDyn pair from the pinned fork commit.
#
# Makes paid model calls (OpenAI) and hosted-service calls (Next Moca) for
# 2 arms x (560 attacked + 60 benign) tasks. Refuses to start unless
# CSBENCH_CONFIRM_PAID_RUN=1. Keys are read from the environment and never
# written to disk or logs:
#   OPENAI_API_KEY, NEXT_MOCA_API_KEY
# Optional: NEXT_MOCA_URL (default https://api.nextmoca.com),
#   NEXT_MOCA_OPERATING_POINT (the published run's operating point is not a
#   listed label; unset, the service uses your account's default),
#   WORK (default ./agent-security-work/agentdyn).
#
# Output: $WORK/runs/{none,next_moca}/... (AgentDojo-format rows) and
# $WORK/results/{summary.json,rows.jsonl}, scored by the same code that
# checks the committed results.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Values come from run_config.json through the csbench package (pip install -e . first).
CSBENCH_PY="${CSBENCH_PY:-python3}"
pin() { "$CSBENCH_PY" -m csbench.suites.agentdyn pin "$1"; }

[ "${CSBENCH_CONFIRM_PAID_RUN:-0}" = 1 ] || { echo "refusing: set CSBENCH_CONFIRM_PAID_RUN=1 to make paid model calls" >&2; exit 2; }
: "${OPENAI_API_KEY:?OPENAI_API_KEY is required}"
: "${NEXT_MOCA_API_KEY:?NEXT_MOCA_API_KEY is required}"

REPO=$(pin fork.repo); COMMIT=$(pin fork.commit)
MODEL=$(pin model.benchmark_flag); VERSION=$(pin benchmark.benchmark_version)
ATTACK=$(pin benchmark.attack); SUITES=$(pin benchmark.suites)
export NEXT_MOCA_URL="${NEXT_MOCA_URL:-$(pin arms.next_moca.service_url)}"

WORK="${WORK:-$PWD/agent-security-work/agentdyn}"
mkdir -p "$WORK"
if [ ! -d "$WORK/fork/.git" ]; then git clone --quiet "$REPO" "$WORK/fork"; fi
git -C "$WORK/fork" fetch --quiet origin "$COMMIT"
git -C "$WORK/fork" checkout --quiet --detach "$COMMIT"
[ "$(git -C "$WORK/fork" rev-parse HEAD)" = "$COMMIT" ] || { echo "fork is not at the pinned commit" >&2; exit 3; }
[ -z "$(git -C "$WORK/fork" status --porcelain)" ] || { echo "fork checkout is dirty" >&2; exit 3; }
if [ ! -x "$WORK/venv/bin/python" ]; then python3 -m venv "$WORK/venv"; fi
"$WORK/venv/bin/pip" install --quiet -e "$WORK/fork"
PY="$WORK/venv/bin/python"

cd "$WORK/fork"
for arm in none next_moca; do
  DEF=(); [ "$arm" = next_moca ] && DEF=(--defense next_moca)
  for suite in $SUITES; do
    for phase in benign attacked; do
      ATT=(); [ "$phase" = attacked ] && ATT=(--attack "$ATTACK")
      echo "== $arm $suite $phase"
      "$PY" -m agentdojo.scripts.benchmark -s "$suite" --model "$MODEL" --benchmark-version "$VERSION" \
        --logdir "$WORK/runs/$arm" ${DEF[@]+"${DEF[@]}"} ${ATT[@]+"${ATT[@]}"}
    done
  done
done

cd "$HERE/../../.."
"$CSBENCH_PY" -m csbench.suites.agentdyn score --none "$WORK/runs/none" --next-moca "$WORK/runs/next_moca" \
  --out "$WORK/results/summary.json"
