import json
import os
from typing import Any, Dict, List

import yaml
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv

from train import (
    PREFIXES,
    add_confirmed_reason,
    analyze_trajectory_differences,
    bloom_failure_reasons,
    build_client,
    build_same_type_examples,
    create_single_game_env,
    format_trajectory_like_prompt,
    load_bloom_memory,
    load_progress,
    load_prompts,
    resolve_prompt_file,
    run_episode,
    save_bloom_memory,
    save_progress,
)


def memory_counts_by_type(memory: Dict[str, List[str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for task_key in PREFIXES.keys():
        counts[task_key] = len(memory.get(task_key, []))
    for k, v in memory.items():
        if k not in counts and isinstance(v, list):
            counts[k] = len(v)
    return counts


def append_block_metrics(
    log_path: str,
    block_start_idx: int,
    block_end_idx: int,
    block_success_sum: float,
    block_count: int,
    block_success_rerun_sum: int,
    block_success_rerun_count: int,
    memory_counts: Dict[str, int],
) -> None:
    block_success_rate = block_success_sum / max(1, block_count)
    avg_rerun_attempts_success = (
        block_success_rerun_sum / block_success_rerun_count if block_success_rerun_count > 0 else 0.0
    )
    record = {
        "block_start_task_idx_1based": block_start_idx + 1,
        "block_end_task_idx_1based": block_end_idx + 1,
        "block_task_count": block_count,
        "block_avg_success_rate": block_success_rate,
        "block_avg_rerun_attempts_on_success": avg_rerun_attempts_success,
        "bloom_memory_counts": memory_counts,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "base_config.yaml"), "r", encoding="utf-8") as reader:
        config = yaml.safe_load(reader)

    split = "train"
    max_tasks = 1000
    root_env = AlfredTWEnv(config, train_eval=split)
    if hasattr(root_env, "collect_game_files"):
        root_env.collect_game_files()
    if hasattr(root_env, "get_game_logic"):
        root_env.get_game_logic()
    env = root_env.init_env(batch_size=1)

    prompts = load_prompts(resolve_prompt_file(base_dir))
    client = build_client()
    model = os.getenv("OPENAI_MODEL", "qwen3-max-2026-01-23")
    bloom_model = os.getenv("BLOOM_MODEL", model)
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "4096"))
    k = int(os.getenv("BLOOM_K", "7"))

    memory_path = os.path.join(base_dir, "bloom_memory.json")
    progress_path = os.path.join(base_dir, "bloom_dist_progress_train_max.json")
    block_log_path = os.path.join(base_dir, "train_max_block_stats.jsonl")

    bloom_memory = load_bloom_memory(memory_path)
    progress = load_progress(progress_path)

    start_idx = int(progress.get("next_idx", 0)) if progress else 0
    rs = progress.get("rs", [0] * 6) if progress else [0] * 6
    cnts = progress.get("cnts", [0] * 6) if progress else [0] * 6
    success_step_sums = progress.get("success_step_sums", [0] * 6) if progress else [0] * 6
    success_step_cnts = progress.get("success_step_cnts", [0] * 6) if progress else [0] * 6

    block_success_sum = float(progress.get("block_success_sum", 0.0)) if progress else 0.0
    block_count = int(progress.get("block_count", 0)) if progress else 0
    block_success_rerun_sum = int(progress.get("block_success_rerun_sum", 0)) if progress else 0
    block_success_rerun_count = int(progress.get("block_success_rerun_count", 0)) if progress else 0
    block_start_idx = int(progress.get("block_start_idx", start_idx)) if progress else start_idx

    if start_idx > 0:
        print(f"resume_from_idx={start_idx}")
        for _ in range(start_idx):
            env.reset()

    idx = start_idx
    try:
        for idx in range(start_idx, max_tasks):
            ob, info = env.reset()
            ob_text = "\n".join(ob[0].split("\n\n")[1:])
            name = "/".join(info["extra.gamefile"][0].split("/")[-3:-1])
            game_file = info["extra.gamefile"][0]

            r = 0.0
            done = False
            rerun_attempts = 0

            for i, (task_key, task_tag) in enumerate(PREFIXES.items()):
                if not name.startswith(task_key):
                    continue

                examples = build_same_type_examples(prompts, task_tag, top_n=2)
                init_prompt = (
                    "Interact with a household to solve a task."
                    + "Here is the task.\n"
                    + ob_text
                    + "\n>"
                )
                r1, done1, traj1 = run_episode(
                    env=env,
                    init_prompt=init_prompt,
                    goal_ob=ob_text,
                    info=info,
                    examples_text=examples,
                    client=client,
                    model=model,
                    max_tokens=max_tokens,
                )
                r, done = r1, done1
                success_traj = traj1

                if not (done1 and r1 > 0):
                    failed_trace = ob_text + "\n" + format_trajectory_like_prompt(traj1)
                    seed_reasons = bloom_memory.get(task_key, [])
                    successful_trace = "None"
                    trajectory_analysis = analyze_trajectory_differences(
                        task_key=task_key,
                        goal_ob=ob_text,
                        examples=examples,
                        successful_trajectory_text=successful_trace,
                        failed_trajectory_text=failed_trace,
                        client=client,
                        model=bloom_model,
                        max_tokens=max_tokens,
                    )
                    reason_candidates = bloom_failure_reasons(
                        task_key=task_key,
                        goal_ob=ob_text,
                        failed_trajectory_text=failed_trace,
                        examples=examples,
                        successful_trajectory_text=successful_trace,
                        trajectory_analysis=trajectory_analysis,
                        seed_reasons=seed_reasons,
                        client=client,
                        model=bloom_model,
                        max_tokens=max_tokens,
                        k=k,
                    )
                    for attempt_idx, reason in enumerate(reason_candidates, start=1):
                        rerun_attempts = attempt_idx
                        sim_env = create_single_game_env(game_file, config)
                        ob2, info2 = sim_env.reset()
                        ob2_text = "\n".join(ob2[0].split("\n\n")[1:])
                        retry_prompt = (
                            "Interact with a household to solve a task."
                            + "Here is the task.\n"
                            + ob2_text
                            + "\n>"
                        )
                        r2, done2, traj2 = run_episode(
                            env=sim_env,
                            init_prompt=retry_prompt,
                            goal_ob=ob2_text,
                            info=info2,
                            examples_text=examples,
                            client=client,
                            model=model,
                            max_tokens=max_tokens,
                            failure_reason=reason,
                            failed_trajectory_text=format_trajectory_like_prompt(traj1),
                        )
                        sim_env.close()
                        if done2 and r2 > 0:
                            add_confirmed_reason(bloom_memory, task_key, reason)
                            save_bloom_memory(memory_path, bloom_memory)
                            r, done = r2, done2
                            success_traj = traj2
                            break

                rs[i] += r
                cnts[i] += 1
                if done and r > 0:
                    step_count = sum(1 for s in success_traj if s.kind == "act")
                    success_step_sums[i] += step_count
                    success_step_cnts[i] += 1
                break

            block_success = 1.0 if (done and r > 0) else 0.0
            block_success_sum += block_success
            block_count += 1
            if block_success > 0:
                block_success_rerun_sum += rerun_attempts
                block_success_rerun_count += 1

            if block_count >= 100:
                append_block_metrics(
                    log_path=block_log_path,
                    block_start_idx=block_start_idx,
                    block_end_idx=idx,
                    block_success_sum=block_success_sum,
                    block_count=block_count,
                    block_success_rerun_sum=block_success_rerun_sum,
                    block_success_rerun_count=block_success_rerun_count,
                    memory_counts=memory_counts_by_type(bloom_memory),
                )
                block_success_sum = 0.0
                block_count = 0
                block_success_rerun_sum = 0
                block_success_rerun_count = 0
                block_start_idx = idx + 1

            save_progress(
                progress_path,
                {
                    "next_idx": idx + 1,
                    "rs": rs,
                    "cnts": cnts,
                    "success_step_sums": success_step_sums,
                    "success_step_cnts": success_step_cnts,
                    "block_success_sum": block_success_sum,
                    "block_count": block_count,
                    "block_success_rerun_sum": block_success_rerun_sum,
                    "block_success_rerun_count": block_success_rerun_count,
                    "block_start_idx": block_start_idx,
                    "memory_path": memory_path,
                    "block_log_path": block_log_path,
                    "k": k,
                    "max_tasks": max_tasks,
                    "split": split,
                },
            )
            print(
                f"task={idx + 1}/{max_tasks} reward={r} done={done} "
                f"rerun_attempts={rerun_attempts} block_count={block_count}"
            )
    except RuntimeError as e:
        print(f"run_interrupted_due_to_llm_error={e}")
        save_progress(
            progress_path,
            {
                "next_idx": idx,
                "rs": rs,
                "cnts": cnts,
                "success_step_sums": success_step_sums,
                "success_step_cnts": success_step_cnts,
                "block_success_sum": block_success_sum,
                "block_count": block_count,
                "block_success_rerun_sum": block_success_rerun_sum,
                "block_success_rerun_count": block_success_rerun_count,
                "block_start_idx": block_start_idx,
                "memory_path": memory_path,
                "block_log_path": block_log_path,
                "k": k,
                "max_tasks": max_tasks,
                "split": split,
            },
        )
        raise

    print("block_stats_file", block_log_path)
    print("progress_file", progress_path)
    print("memory_file", memory_path)


if __name__ == "__main__":
    main()
