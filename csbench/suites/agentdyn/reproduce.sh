#!/usr/bin/env bash
# Reproduce the published GPT-4o AgentDyn pair from the pinned fork commit.
#
# Makes paid model calls (OpenAI) and hosted-service calls (Next Moca) for
# 2 arms x (560 attacked + 60 benign) tasks. Refuses to start unless
# CSBENCH_CONFIRM_PAID_RUN=1. Keys are read from the environment and never
# written to disk or logs:
#   OPENAI_API_KEY, NEXT_MOCA_API_KEY
#   NEXT_MOCA_OPERATING_POINT: required. The published run used an operating
#   point that is not a listed label, so it cannot be selected here; name a
#   listed label explicitly. (Left unset, the pinned element would fall back to
#   its own built-in default, which is not the published point either.)
# Optional: NEXT_MOCA_URL (default https://api.nextmoca.com),
#   WORK (default ./agent-security-work/agentdyn).
#
# Output: $WORK/run-<utc timestamp>/runs/{none,next_moca}/... (AgentDojo-format rows) and
# .../results/{summary.json,rows.jsonl}, scored by the same code that
# checks the committed results.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Values come from run_config.json through the csbench package (pip install -e . first).
CSBENCH_PY="${CSBENCH_PY:-python3}"
pin() { "$CSBENCH_PY" -m csbench.suites.agentdyn pin "$1"; }

[ "${CSBENCH_CONFIRM_PAID_RUN:-0}" = 1 ] || { echo "refusing: set CSBENCH_CONFIRM_PAID_RUN=1 to make paid model calls" >&2; exit 2; }
: "${OPENAI_API_KEY:?OPENAI_API_KEY is required}"
: "${NEXT_MOCA_API_KEY:?NEXT_MOCA_API_KEY is required}"
: "${NEXT_MOCA_OPERATING_POINT:?NEXT_MOCA_OPERATING_POINT is required: name a listed operating point}"
export NEXT_MOCA_OPERATING_POINT

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

# A fresh run root per invocation: the benchmark scripts skip any row that
# already exists, so reusing a directory would silently mix runs.
RUN="$WORK/run-$(date -u +%Y%m%dT%H%M%SZ)"
[ ! -e "$RUN" ] || { echo "$RUN already exists" >&2; exit 3; }
mkdir -p "$RUN"
echo "run root $RUN"

# The service's build id, read at the start and the end of the run; the
# summary records it, and a run that spans two builds is not scored.
health_build() {
  "$PY" -c 'import json, os, urllib.request
req = urllib.request.Request(os.environ["NEXT_MOCA_URL"].rstrip("/") + "/v1/health",
                             headers={"Authorization": "Bearer " + os.environ["NEXT_MOCA_API_KEY"]})
print(json.load(urllib.request.urlopen(req, timeout=20)).get("build_id") or "")'
}
BUILD_START=$(health_build)
[ -n "$BUILD_START" ] || { echo "the service reported no build id" >&2; exit 4; }
echo "service build $BUILD_START"

cd "$WORK/fork"
for arm in none next_moca; do
  DEF=(); [ "$arm" = next_moca ] && DEF=(--defense next_moca)
  for suite in $SUITES; do
    for phase in benign attacked; do
      ATT=(); [ "$phase" = attacked ] && ATT=(--attack "$ATTACK")
      echo "== $arm $suite $phase"
      "$PY" -m agentdojo.scripts.benchmark -s "$suite" --model "$MODEL" --benchmark-version "$VERSION" \
        --logdir "$RUN/runs/$arm" ${DEF[@]+"${DEF[@]}"} ${ATT[@]+"${ATT[@]}"}
    done
  done
done

BUILD_END=$(health_build)
[ "$BUILD_START" = "$BUILD_END" ] || { echo "service build changed during the run ($BUILD_START -> $BUILD_END): not scored" >&2; exit 6; }

cd "$HERE/../../.."
"$CSBENCH_PY" -m csbench.suites.agentdyn score --none "$RUN/runs/none" --next-moca "$RUN/runs/next_moca" \
  --out "$RUN/results/summary.json" --build-id "$BUILD_START"
