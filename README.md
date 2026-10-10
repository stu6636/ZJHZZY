# BloomGoal

Code for BloomGoal in ALFWorld, WebShop, and InterCodeSQL. This repository contains BloomGoal only, plus BloomGoal-Retrieval for InterCodeSQL. Benchmark datasets, databases, server code, experiment logs, and API credentials are not included.

## Methods and files

| Environment | Method | Entry point |
| --- | --- | --- |
| ALFWorld | BloomGoal | `bloom_dist.py` (`bloom_dist_zero_shot.py` for zero-shot) |
| WebShop | BloomGoal | `webshop/bloomgoal_webshop_runner.py` |
| InterCodeSQL | BloomGoal, FIFO lesson window of five | `intercode_sql/sqlite_sql/bloomgoal_sqlite_runner.py` |
| InterCodeSQL | BloomGoal-Retrieval, persistent lessons | `intercode_sql/sqlite_sql/bloomgoal_retrieval_sqlite_runner.py` |

The WebShop runner keeps the latest 10 action-scoring and reflection-scoring lessons. The InterCodeSQL BloomGoal runner keeps the latest five of each. BloomGoal-Retrieval retains historical lessons, retrieves the five most relevant action-scoring lessons with TF-IDF cosine similarity, and uses a recent subset of reflection-scoring lessons in its reflection scorer. The retrieval variant is provided for InterCodeSQL only.

## API configuration

The WebShop and InterCodeSQL runners use an OpenAI-compatible chat-completions endpoint. Set `OPENAI_API_KEY` in your environment. `OPENAI_MODEL` defaults to `gpt-5-mini`, and `OPENAI_BASE_URL` defaults to `https://api.openai.com/v1`. Set these variables to your provider's model and endpoint when needed. Do not commit credentials or `.api_env` files.

The existing ALFWorld scripts read `API_KEY.txt`, which is a tracked empty placeholder in this repository. Keep real credentials out of commits. The ALFWorld implementation is unchanged.

## ALFWorld

Install the [ALFWorld benchmark](https://github.com/alfworld/alfworld) and its dependencies, then install this repository's ALFWorld requirements:

```bash
pip install -r requirement.txt
python bloom_dist.py
```

The ALFWorld script writes `bloom_dist_progress.json` and `bloom_memory.json`. Use `python train.py` or `python train_max.py` for training before evaluation when applicable. The zero-shot entry point is `python bloom_dist_zero_shot.py`.

## WebShop BloomGoal

Install the Python dependencies with `pip install -r webshop/requirements.txt`. Set up and start the [official WebShop server](https://github.com/princeton-nlp/WebShop) separately. By default, the runner connects to `http://127.0.0.1:3000`; override this with `WEBSHOP_URL` if needed. The local `webshop/webshop_environment.py` adapter handles page observations and actions, but does not include the WebShop server or product data.

Run one benchmark session from the repository root:

```bash
python webshop/bloomgoal_webshop_runner.py --session example_session
```

The paper defaults are 50 total action steps, at most two retries after the initial attempt, and three candidate actions per step (`--max-steps 50 --retry-k 2 --max-candidates 3`). Use separate `--out-dir` and `--memory-path` values for independent evaluation runs. A shared memory path deliberately carries lessons between tasks.

## InterCodeSQL BloomGoal and BloomGoal-Retrieval

Install `pip install -r intercode_sql/requirements.txt`. These runners use a local SQLite-backed adaptation of [InterCodeSQL](https://github.com/princeton-nlp/intercode), not its official Docker/MySQL container. Supply a JSON task list whose entries contain `db`, `query`, and `gold`, and corresponding SQLite databases under `--sqlite-root`. A database can be located at either `<sqlite-root>/<db>/<db>.sqlite` or `<sqlite-root>/<db>.sqlite` (the `.db` extension also works). The environment copies each database to a temporary directory for every attempt.

From the `intercode_sql` directory, run either method:

```bash
python -m sqlite_sql.bloomgoal_sqlite_runner --data-path path/to/tasks.json --sqlite-root path/to/databases --task-index 0 --limit 1
python -m sqlite_sql.bloomgoal_retrieval_sqlite_runner --data-path path/to/tasks.json --sqlite-root path/to/databases --task-index 0 --limit 1
```

Both default to 15 turns per attempt, two retries after the initial attempt, and eight candidate actions per step (`--max-turns 15 --retry-k 2 --max-candidates 8`). Each run creates its own result directory and memory file unless `--out-dir` or `--memory-path` is specified. Set an explicit shared memory path only when you intend to reuse lessons across tasks. Keep the base and retrieval memory files separate.

## Verification

The offline tests require the WebShop and InterCodeSQL Python dependencies but no live model API, WebShop server, or benchmark data:

```bash
python -m unittest discover -s tests -v
```

They check runner imports, paper-default parameters, WebShop navigation, SQLite execution and reward, action parsing, and the distinct lesson-storage policies. They do not reproduce the paper's reported success rates; that requires the original task splits, server/database assets, and model access.
