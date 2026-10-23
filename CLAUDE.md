# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Official implementation of **ExpeL: LLM Agents are Experiential Learners** (AAAI 2024). An LLM agent gathers experience on training tasks, distills natural-language "insights" (rules) from it, and at evaluation time uses those rules plus retrieved similar successful trajectories as few-shots. There is no fine-tuning; all "learning" lives in prompts.

Benchmarks: `hotpotqa`, `fever`, `alfworld`, `webshop`. The code is pinned to old libraries (`langchain==0.0.181`, `openai==0.27.7`, Python 3.9.17); newer versions of either will break imports such as `langchain.chat_models`, `openai.error`.

## Commands

There is no build step, linter, or test suite. Everything is run through three Hydra entry points from the repo root; config groups are `benchmark=` and `agent=` and any value can be overridden on the command line (`key=value`).

```bash
pip install -r requirements.txt
# needs OPENAI_API_KEY (in .env or env var), otherwise it prompts via getpass

# 1. Experience gathering (ReAct + Reflexion on training tasks)
python train.py benchmark=<hotpotqa|fever|alfworld|webshop> run_name=<train-run> testing=false resume=false

# 2. Insight extraction (builds the rule list, per fold)
python insight_extraction.py benchmark=<b> load_run_name=<train-run> run_name=<insights-run> \
  agent.llm=gpt-4 agent.max_num_rules=10 agent.success_critique_num=8 testing=false resume=false

# 3. Evaluation (rules + task-similarity few-shot retrieval)
python eval.py benchmark=<b> load_run_name=extracted_insights/<insights-run> run_name=<eval-run> \
  agent.fewshot_strategy=task_similarity agent.retrieval_kwargs.max_fewshot_tokens=auto testing=false resume=false
```

Paper-faithful command lines per benchmark are in `README.md`. Notable per-benchmark differences: WebShop eval also sets `agent.retrieval_kwargs.buffer_retrieve_ratio=20`; HotpotQA eval omits `max_fewshot_tokens`.

- **Cheap smoke test:** `testing=true` (the default in every config) makes no API calls. `prompt_critique` prints the prompt and reads the LLM's answer from `input()`, and the API key is replaced with a placeholder. The default `run_name=test` also skips the overwrite prompt.
- **Resuming:** `resume=true` on `train.py`/`eval.py`. For `insight_extraction.py` you must also pass `resume_fold=<k>`, and drop `load_run_name` (see README).
- **Outputs:** `logs/<benchmark>/expel/<run_name>.pkl` for training, `.../extracted_insights/<run_name>.pkl` (plus `fold_<k>/`) for insights, `.../eval/<run_name>.pkl` for eval. Change the root with `log_dir=`. Each run writes a human log (`log`), a verbose `true_log`, and a list of agent state `dicts`. `tools/visualize_logs.py` (config `configs/visualize_logs.yaml`) plots logs.
- **External setup:** ALFWorld needs `pip install alfworld[full]` and `alfworld-download` into `data/alfworld`. The committed `data/alfworld/{json_2.1.1,logic}` directories were deleted, but `configs/benchmark/alfworld.yaml` still references them. WebShop needs its own server running separately, with its URL set as `WEBSHOP_URL` in `envs/webshop/webshop.py`.

## Architecture

### Agent hierarchy (`agent/`)
`BaseAgent` → `ReactAgent` (think/act/observe loop against an env) → `ReflectAgent` (on failure, writes a reflection and retries, up to `agent.max_reflection_depth` trials) → `ExpelAgent` (adds critique/rule extraction and retrieval-based few-shots). `agent.AGENT` maps config `agent.name` (`react`/`reflection`/`expel`) to the class. All three scripts build the agent with the same long keyword list, most of it benchmark-specific callables and prompt templates looked up by `cfg.benchmark.name`. **A new constructor argument has to be added in `train.py`, `insight_extraction.py` and `eval.py` consistently.**

### Per-benchmark plug-ins
Almost everything benchmark-specific is a dict keyed by benchmark name, so adding or changing a benchmark means touching several files:
- `envs/__init__.py`: `ENVS` (env class) and `INIT_TASKS_FN` (builds the task list from the cfg and the task file).
- `prompts/__init__.py`: `SYSTEM_INSTRUCTION`, `FEWSHOTS`, `REFLECTION_FEWSHOTS`, `RULE_TEMPLATE`, `LLM_PARSER`, `OBSERVATION_FORMATTER`, `STEP_IDENTIFIER`, `CYCLER`/`STEP_CYCLER`, `STEP_STRIPPER`, etc., built from `prompts/<benchmark>.py` and `prompts/templates/`.
- `configs/benchmark/<name>.yaml`: `task_file`, `max_steps`, `num_fewshots`, `eval_configs.k_folds`.
- `utils.py`: `get_env_name_from_task`, `get_fewshot_max_tokens`, and benchmark-specific result functions.

### The ExpeL pipeline across files
1. `train.py` runs `agent.run(mode='train')` per task. `ExpelAgent.next_task` files each finished `Trajectory` (`memory/episode.py`) into `succeeded_trial_history[task]` or `failed_trial_history[task]`. Whole agent state is pickled each task into `dicts`.
2. `insight_extraction.py` reloads `dicts[-1]` into a fresh agent via `load_checkpoint(..., no_load_list=...)`. The big `no_load_list` decides which saved attributes are overwritten and which come from the current config, so add new stateful attributes there deliberately. It then splits tasks into `k_folds` (`get_split_eval_idx_list`). For each fold it calls `ExpelAgent.create_rules` on the training tasks of that fold: compare success vs. failure → batched success critiques → rule edits. The LLM emits `ADD/EDIT/REMOVE/AGREE` lines, parsed by `parse_rules` and applied by `update_rules` (module-level functions at the bottom of `agent/expel.py`). Rules carry vote counts and are dropped at 0 (`ADD` +2, `AGREE`/`EDIT` +1, `REMOVE` −1, or −3 when the list is full). Resume state (`critique_summary_section/idx/fold`) is stored in the saved dict.
3. `eval.py` iterates folds. At each fold start, rules are either loaded from the per-fold cache saved during insight extraction (`create_rules(load_cache_fold=k)`, when `load_cache_rules=true`, the default) or regenerated. `no_rules=true` disables rule injection (few-shot-only ablation). Then `ExpelAgent.run(mode='eval', eval_idx=...)` runs the held-out tasks. `agent.eval()` / `agent.train()` toggles `_train`, which changes prompt assembly: in eval, `insert_before_task_prompt` injects the rules, and `update_dynamic_prompt_components` rebuilds a FAISS vectorstore (`setup_vectorstore`) over successful trajectories plus the original few-shots, retrieves by `fewshot_strategy` and optional `reranker`, and swaps them into `prompt_history`. Few-shots longer than `max_fewshot_tokens` or equal to the current task are skipped.

Important consequence: evaluation tasks are the held-out fold of the *same* task list used for training, so keep fold indices (`eval_idx_list`, stored in the saved dict) consistent between insight extraction and eval.

### Misc
- `models/llm.py`: `GPTWrapper` only accepts model names containing `gpt` and retries 6 times on `RateLimitError`. `ReactAgent` builds both `llm` and `long_context_llm` (`gpt-3.5-turbo-16k`); the latter is used for agent/reflection prompts and as the critique fallback on `InvalidRequestError`.
- `memory/__init__.py`: embedder (`huggingface` default, `all-mpnet-base-v2`, also `openai`/`llama`/`gpt4all`) and retriever selection.
- `utils.py` holds checkpoint save/load, log splitting, stats plotting, and the EM metric. The scripts parse logs back with regexes (`split_logs_by_task`), so changing log formatting in `agent/` can break result reporting.
