# Datasets

Upstream datasets are **not** redistributed in this repository. Each suite is
obtained with a fetch-script that downloads from the canonical upstream source
under that source's own license. Fetched data lands in a git-ignored `data/`
directory and is never committed.

Fetch-scripts and per-suite license notes live alongside each suite's
harness. v1 suites:

| Suite | Upstream | License | Redistribute raw data? |
|---|---|---|---|
| RULER | NVIDIA/RULER generator | Apache-2.0 (code); downstream corpora carry their own licenses | No |
| Function-calling tool suite | Berkeley Function-Calling Leaderboard | Apache-2.0 | No (fetch-script) |
| SQuAD v2 | `rajpurkar/squad_v2` | CC BY-SA 4.0 | No (share-alike) |
| TruthfulQA | `truthfulqa/truthful_qa` (multiple_choice) | Apache-2.0 | No (fetch-script) |
| GSM8K / GSM-IC (diagnostic) | `openai/gsm8k` | MIT | No (fetch-script) |

Each fetch-script pins an upstream revision so a reproduction is deterministic.
Nothing in this repository relicenses upstream data.

## RULER

`python scripts/fetch_ruler.py --out data/ruler` clones NVIDIA/RULER at a pinned
commit, downloads the upstream corpora, and generates
`data/ruler/<length>/<task>/test.jsonl`. It needs `git`/`git-lfs`, `wget`, and a
`GEMINI_API_KEY` (RULER's Gemini tokenizer calls the `countTokens` API for exact
haystack sizing). See the script header for the per-corpus license terms -
generated data is produced locally under those terms and is not redistributed.
