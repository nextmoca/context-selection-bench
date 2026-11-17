#!/usr/bin/env bash
# Reproduce the published GPT-4o AutoDojo banking pair from the pinned fork commit.
#
# Two stages per arm: optimize injections against that arm's own pipeline,
# then benchmark the optimized injections (variant 0) and the benign tasks.
# Makes paid model calls (OpenAI target, Gemini optimizer) and hosted-service
# calls (Next Moca). Refuses to start unless CSBENCH_CONFIRM_PAID_RUN=1. Keys
# are read from the environment and never written to disk or logs:
#   OPENAI_API_KEY, GEMINI_API_KEY, NEXT_MOCA_API_KEY
# Optional: NEXT_MOCA_URL (default https://api.nextmoca.com),
#   NEXT_MOCA_OPERATING_POINT (default: the value in run_config.json),
#   WORK (default ./agent-security-work/autodojo).
#
# The optimizer is stochastic: a fresh run produces a different injection
# cache, so attack success is expected to vary from the published run.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Values come from run_config.json through the csbench package (pip install -e . first).
CSBENCH_PY="${CSBENCH_PY:-python3}"
pin() { "$CSBENCH_PY" -m csbench.suites.autodojo pin "$1"; }

[ "${CSBENCH_CONFIRM_PAID_RUN:-0}" = 1 ] || { echo "refusing: set CSBENCH_CONFIRM_PAID_RUN=1 to make paid model calls" >&2; exit 2; }
: "${OPENAI_API_KEY:?OPENAI_API_KEY is required}"
: "${GEMINI_API_KEY:?GEMINI_API_KEY is required}"
: "${NEXT_MOCA_API_KEY:?NEXT_MOCA_API_KEY is required}"

REPO=$(pin fork.repo); COMMIT=$(pin fork.commit)
TARGET=$(pin model.benchmark_flag); VERSION=$(pin benchmark.benchmark_version)
SUITE=$(pin benchmark.suites)
OPT=$(pin benchmark.optimizer.model); OPT_PROVIDER=$(pin benchmark.optimizer.provider)
ITER=$(pin benchmark.optimizer.iterations); NV=$(pin benchmark.optimizer.n_variants)
VARIANT=$(pin benchmark.optimizer.benchmarked_variant); CONC=$(pin benchmark.optimizer.eval_concurrency)
ANALYZER=$(pin benchmark.optimizer.analyzer_prompt); INJ_PROMPT=$(pin benchmark.optimizer.injection_prompt)
export NEXT_MOCA_URL="${NEXT_MOCA_URL:-$(pin arms.next_moca.service_url)}"
export NEXT_MOCA_OPERATING_POINT="${NEXT_MOCA_OPERATING_POINT:-$(pin arms.next_moca.operating_point)}"

WORK="${WORK:-$PWD/agent-security-work/autodojo}"
mkdir -p "$WORK"
if [ ! -d "$WORK/fork/.git" ]; then git clone --quiet "$REPO" "$WORK/fork"; fi
git -C "$WORK/fork" fetch --quiet origin "$COMMIT"
git -C "$WORK/fork" checkout --quiet --detach "$COMMIT"
[ "$(git -C "$WORK/fork" rev-parse HEAD)" = "$COMMIT" ] || { echo "fork is not at the pinned commit" >&2; exit 3; }
[ -z "$(git -C "$WORK/fork" status --porcelain)" ] || { echo "fork checkout is dirty" >&2; exit 3; }
if [ ! -x "$WORK/venv/bin/python" ]; then python3 -m venv "$WORK/venv"; fi
"$WORK/venv/bin/pip" install --quiet -e "$WORK/fork/agentdojo" json_repair nltk
PY="$WORK/venv/bin/python"

F="$WORK/fork"
export AGENTDOJO_DEFENSE_PLUGINS=np_autodojo_plugin NP_OPENAI_DIRECT=1
export PYTHONPATH="$F/next_moca:$F/agentdojo/src:$F/agentdojo/variant_generation"
export AUTODOJO_OUTPUT_DIR="$WORK/variants"
cd "$F"

for arm in next_moca none; do
  DEF=(); [ "$arm" = next_moca ] && DEF=(--defense next_moca --run-defense)
  echo "== optimize $arm"
  "$PY" agentdojo/variant_generation/optimize_variants.py --suite "$SUITE" --n-variants "$NV" --iterations "$ITER" \
    --eval-asr --target-model "$TARGET" --model "$OPT" --provider "$OPT_PROVIDER" \
    --analyzer-prompt "$ANALYZER" --injection-prompt "$INJ_PROMPT" \
    --parallel-eval --eval-concurrency "$CONC" --store-traces ${DEF[@]+"${DEF[@]}"} 2>&1 | tee "$WORK/opt-$arm.log"
  # The optimizer reports some provider failures only in its log; a cache built
  # through them is not a valid attack, so stop rather than benchmark it.
  if grep -qE 'API Error|Max retries' "$WORK/opt-$arm.log"; then echo "optimizer API errors for $arm: stopping" >&2; exit 4; fi
done

for arm in next_moca none; do
  DEF=(); CACHE_DIR=no_defense
  [ "$arm" = next_moca ] && { DEF=(--defense next_moca); CACHE_DIR=next_moca; }
  CACHE="$WORK/variants/$SUITE/$TARGET/$CACHE_DIR/injections.json"
  [ -f "$CACHE" ] || { echo "missing optimized cache $CACHE" >&2; exit 5; }
  echo "== benchmark $arm benign"
  "$PY" -m agentdojo.scripts.benchmark --model "$TARGET" --suite "$SUITE" --benchmark-version "$VERSION" \
    --logdir "$WORK/runs/$arm" ${DEF[@]+"${DEF[@]}"}
  echo "== benchmark $arm attacked"
  AUTODOJO_CACHE="$CACHE" AUTODOJO_VARIANT="$VARIANT" "$PY" -m agentdojo.scripts.benchmark --model "$TARGET" \
    --suite "$SUITE" --benchmark-version "$VERSION" --logdir "$WORK/runs/$arm" ${DEF[@]+"${DEF[@]}"} --attack autodojo
done

cd "$HERE/../../.."
"$CSBENCH_PY" -m csbench.suites.autodojo score --none "$WORK/runs/none" --next-moca "$WORK/runs/next_moca" \
  --out "$WORK/results/summary.json"
