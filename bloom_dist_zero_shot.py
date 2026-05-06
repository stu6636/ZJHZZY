import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import yaml
import textworld
import textworld.gym
from openai import OpenAI
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv, AlfredDemangler, AlfredInfos

PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}


ACTION_SPACE_INTRO = (
    "Environment action space introduction:\n"
    "goto {recep}: Navigate to the specified receptacle (cabinet/sink/table etc.)\n"
    "open {recep}: Open the target receptacle to access inner objects\n"
    "close {recep}: Close an opened receptacle\n"
    "take {obj} from {recep}: Pick up the object from the receptacle to inventory\n"
    "put {obj} in/on {recep}: Place the inventory object on/in the receptacle\n"
    "clean {obj} with {recep}: Clean the object using a water-based receptacle (sink/basin)\n"
    "heat {obj} with {recep}: Heat the object using a heating receptacle (microwave/stove)\n"
    "cool {obj} with {recep}: Cool the object using a cooling receptacle (fridge)\n"
    "toggle {obj/recep}: Turn on/off a functional object/receptacle (lamp/microwave)\n"
    "inventory: Check the objects the agent is currently carrying\n"
    "examine: Inspect an object/receptacle; confirm object state (clean/hot/cool etc.)\n"
    "Failure feedback: \"Nothing happens\" for invalid operations\n"
)


@dataclass
class StepRecord:
    kind: str
    text: str

def resolve_prompt_file(base_dir: str) -> str:
    candidates = [
        os.getenv("ALFWORLD_PROMPT_FILE", ""),
        os.path.join(base_dir, "prompts", "alfworld_3prompts.json"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError("Cannot find prompt file")


def load_prompts(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in data.items():
        if isinstance(k, str) and isinstance(v, str):
            out[k] = v
    return out

with open("API_KEY.txt", "r", encoding="utf-8") as f:
    API_KEY = f.read().strip()

client = OpenAI(
    api_key=API_KEY,  # your API Key
    base_url="https://api.chatanywhere.tech"
)
def build_client() -> OpenAI:
    return client


def _llm_request_with_backoff(
    prompt: str,
    client: OpenAI,
    model: str,
    max_tokens: int,
    stop_newline: bool = False,
) -> str:
    max_fail_before_pause = 3
    max_total_fail = 6
    pause_seconds = 20 * 60
    retry = 0
    while retry < max_total_fail:
        try:
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": max_tokens,
            }
            if stop_newline:
                kwargs["stop"] = "\n"
            response = client.chat.completions.create(**kwargs)
            text = (response.choices[0].message.content or "").strip()
            if stop_newline:
                return text.split("\n")[0].strip()
            return text
        except Exception as e:
            retry += 1
            print(f"llm_call_failed retry={retry}/{max_total_fail} err={e}")
            if retry == max_fail_before_pause:
                print(f"llm_call_failed_3_times sleep_seconds={pause_seconds}")
                time.sleep(pause_seconds)
                continue
            if retry >= max_total_fail:
                raise RuntimeError(f"LLM failed after {max_total_fail} retries")
            time.sleep(2)
    raise RuntimeError(f"LLM failed after {max_total_fail} retries")


def llm2(prompt: str, client: OpenAI, model: str, max_tokens: int) -> str:
    return _llm_request_with_backoff(
        prompt=prompt,
        client=client,
        model=model,
        max_tokens=max_tokens,
        stop_newline=False,
    )


def llm(prompt: str, client: OpenAI, model: str, max_tokens: int) -> str:
    return _llm_request_with_backoff(
        prompt=prompt,
        client=client,
        model=model,
        max_tokens=max_tokens,
        stop_newline=True,
    )


def process_ob(ob):
    if isinstance(ob, list):
        ob = ob[0] if ob else ""
    if isinstance(ob, tuple):
        ob = ob[0] if ob else ""
    if ob is None:
        ob = ""
    if not isinstance(ob, str):
        ob = str(ob)
    if ob.startswith("You arrive at loc "):
        ob = ob[ob.find(". ") + 2 :]
    return ob


def get_admissible(info: Dict[str, Any]) -> List[str]:
    if not isinstance(info, dict):
        return []
    raw = info.get("admissible_commands")
    if isinstance(raw, list) and raw:
        first = raw[0]
        if isinstance(first, list):
            return [a for a in first if isinstance(a, str)]
    return []


def parse_action(output: str) -> Tuple[str, str]:
    text = (output or "").strip()
    if text.startswith(">"):
        text = text[1:].strip()
    if text.lower().startswith("act:"):
        text = text[4:].strip()
    if text.lower().startswith("think:"):
        return "think", text[6:].strip()
    return "act", text


def format_trajectory_like_prompt(traj: List[StepRecord]) -> str:
    lines: List[str] = []
    for s in traj:
        if s.kind == "think":
            lines.append(f"> think: {s.text}\nOK.")
        elif s.kind == "act":
            lines.append(f"> {s.text}")
        elif s.kind == "ob":
            lines.append(s.text)
    return "\n".join(lines)


def build_same_type_examples(prompts: Dict[str, str], task_tag: str, top_n: int = 3) -> str:
    keys = [k for k in sorted(prompts.keys()) if k.startswith(f"react_{task_tag}_")]
    selected = keys[:top_n]
    if not selected:
        return "None"
    return "\n\n".join([f"Successful example {i + 1}:\n{prompts[k]}" for i, k in enumerate(selected)])


def load_bloom_memory(path: str) -> Dict[str, List[str]]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            out: Dict[str, List[str]] = {}
            for k, v in data.items():
                if isinstance(k, str) and isinstance(v, list):
                    out[k] = [str(x).strip() for x in v if str(x).strip()]
            return out
    except Exception:
        pass
    return {}


def save_bloom_memory(path: str, memory: Dict[str, List[str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


def add_confirmed_reason(memory: Dict[str, List[str]], task_key: str, reason: str, max_per_type: int = 50) -> None:
    reason = reason.strip()
    if not reason:
        return
    arr = memory.get(task_key, [])
    if reason in arr:
        arr.remove(reason)
    arr.insert(0, reason)
    memory[task_key] = arr[:max_per_type]


def load_progress(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_progress(path: str, progress: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def _max_episode_steps(config: Dict[str, Any]) -> int:
    training_method = config["general"]["training_method"]
    if training_method == "dqn":
        return config["rl"]["training"]["max_nb_steps_per_episode"]
    if training_method == "dagger":
        return config["dagger"]["training"]["max_nb_steps_per_episode"]
    return 60


def create_single_game_env(game_file: str, config: Dict[str, Any]):
    alfred_demangler = AlfredDemangler(shuffle=False)
    wrappers_tw = [alfred_demangler, AlfredInfos]
    request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
    env_id = textworld.gym.register_games(
        [game_file],
        request_infos,
        batch_size=1,
        asynchronous=False,
        max_episode_steps=_max_episode_steps(config),
        wrappers=wrappers_tw,
    )
    return textworld.gym.make(env_id)


def extract_goal_entities(goal_ob: str) -> List[str]:
    goal_line = goal_ob.strip().split("\n")[0].lower()
    tokens = re.findall(r"[a-z][a-z0-9_]*", goal_line)
    stopwords = {
        "your", "task", "is", "to", "the", "a", "an", "and", "or", "then", "with", "from", "into", "in", "on", "at",
        "for", "of", "some", "all", "one", "two", "three", "put", "place", "pick", "clean", "heat", "cool", "look",
        "examine", "find", "move", "go", "open", "close", "toggle", "inventory", "object", "objects",
    }
    out: List[str] = []
    for t in tokens:
        if t in stopwords:
            continue
        if t.isdigit():
            continue
        if t not in out:
            out.append(t)
    return out


def extract_action_entities(action: str) -> List[str]:
    text = action.lower()
    text = re.sub(r"\b\d+\b", " ", text)
    tokens = re.findall(r"[a-z][a-z0-9_]*", text)
    stopwords = {
        "go", "to", "goto", "open", "close", "take", "put", "in", "on", "with", "from", "toggle", "inventory", "examine",
    }
    out: List[str] = []
    for t in tokens:
        if t in stopwords:
            continue
        if t not in out:
            out.append(t)
    return out


def lexical_reduction(goal_ob: str, action: str) -> float:
    goal_entities = extract_goal_entities(goal_ob)
    if not goal_entities:
        return 0.0
    action_entities = set(extract_action_entities(action))
    if not action_entities:
        return 0.0
    matches = sum(1 for g in goal_entities if g in action_entities)
    ratio = matches / max(1, len(set(goal_entities)))
    score = 0.5 * ratio
    if score < 0.0:
        return 0.0
    if score > 0.5:
        return 0.5
    return score


def parse_action_distance_json(raw: str, admissible: List[str]) -> Dict[str, float]:
    result = {a: 0.0 for a in admissible}
    text = raw.strip()
    if not text:
        return result
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                for a in admissible:
                    if a in obj:
                        try:
                            v = float(obj[a])
                        except Exception:
                            v = 0.0
                        if v < 0.0:
                            v = 0.0
                        if v > 0.5:
                            v = 0.5
                        result[a] = v
                return result
    except Exception:
        pass
    return result


def llm_action_reduction_scores(
    admissible: List[str],
    examples_text: str,
    goal_ob: str,
    trajectory_text: str,
    client: OpenAI,
    model: str,
    max_tokens: int,
) -> Dict[str, float]:
    if not admissible:
        return {}
    actions_block = "\n".join([f"{i + 1}. {a}" for i, a in enumerate(admissible)])
    prompt = (
        "You are an ALFWorld action distance estimator.\n"
        "For each admissible action, estimate a reduction score in [0, 0.5].\n"
        "Smaller distance to task completion => larger reduction score.\n"
        f"Successful examples:\n{examples_text}\n"
        f"Current task goal:\n{goal_ob}\n"
        f"Current trajectory:\n{trajectory_text}\n"
        f"Admissible actions:\n{actions_block}\n"
        "Output JSON object only. Keys must be exact action strings and values must be numbers in [0,0.5].\n"
    )
    out = llm2(prompt, client=client, model=model, max_tokens=max_tokens)
    return parse_action_distance_json(out, admissible)


def compute_action_distances(
    admissible: List[str],
    examples_text: str,
    goal_ob: str,
    trajectory: List[StepRecord],
    client: OpenAI,
    model: str,
    max_tokens: int,
) -> Dict[str, float]:
    traj_text = format_trajectory_like_prompt(trajectory)
    llm_scores = llm_action_reduction_scores(
        admissible=admissible,
        examples_text=examples_text,
        goal_ob=goal_ob,
        trajectory_text=traj_text,
        client=client,
        model=model,
        max_tokens=max_tokens,
    )
    out: Dict[str, float] = {}
    for a in admissible:
        lexical_score = lexical_reduction(goal_ob, a)
        llm_score = llm_scores.get(a, 0.0)
        total_reduction = 0.5*lexical_score + llm_score
        if total_reduction > 1.0:
            total_reduction = 1.0
        if total_reduction < 0.0:
            total_reduction = 0.0
        distance = 1.0 - total_reduction
        if distance < 0.0:
            distance = 0.0
        if distance > 1.0:
            distance = 1.0
        out[a] = round(distance, 4)
    return out


def build_admissible_with_distance(admissible: List[str], distances: Dict[str, float]) -> str:
    lines = []
    for i, a in enumerate(admissible):
        d = distances.get(a, 1.0)
        lines.append(f"{i + 1}. {a} | distance={d:.4f}")
    return "\n".join(lines)


def run_episode(
    env,
    init_prompt: str,
    goal_ob: str,
    info: Dict[str, Any],
    examples_text: str,
    client: OpenAI,
    model: str,
    max_tokens: int,
    failure_reason: str = "",
    failed_trajectory_text: str = "",
    max_steps: int = 60,
) -> Tuple[float, bool, List[StepRecord]]:
    prompt = ""
    trajectory: List[StepRecord] = []
    reason_block = ""
    if failure_reason:
        reason_block = (
            f"\nPrevious failed trajectory:\n{failed_trajectory_text}\n"
            f"\nHypothesized failure reason: {failure_reason}\n"
            "Use this reason to avoid repeating the same failure.\n"
            f"{goal_ob}\n>"
        )
    print("failure_reason",failure_reason)
    for _ in range(1, max_steps + 1):
        admissible = get_admissible(info)
        action_distances = compute_action_distances(
            admissible=admissible,
            examples_text=examples_text,
            goal_ob=goal_ob,
            trajectory=trajectory,
            client=client,
            model=model,
            max_tokens=max_tokens,
        )
        admissible_block = build_admissible_with_distance(admissible, action_distances)
        think_prompt = (
            init_prompt
            + reason_block
            + prompt
            + f"\nAdmissible actions with distance to goal:\n{admissible_block}\n"
            + "Output exactly one line starting with 'think:'.Coutain your short plan for the next steps.Your thought will guide the 'action'."
        )
        think_output = llm2(think_prompt, client=client, model=model, max_tokens=max_tokens).strip()
        _, think_text = parse_action(think_output)
        trajectory.append(StepRecord(kind="think", text=think_text))
        prompt += f" think: {think_text}\nOK.\n>"
        #print(f" think: {think_text}\nOK.\n>")
        act_prompt = (
            init_prompt
            + reason_block
            + prompt
            + f"\nAdmissible actions with distance to goal:\n{admissible_block}\n"
            + "Output exactly one line and it must be one admissible action."
        )
        act_output = llm(act_prompt, client=client, model=model, max_tokens=max_tokens).strip()
        _, action_text = parse_action(act_output)
        trajectory.append(StepRecord(kind="act", text=action_text))
        observation, reward, done, info = env.step([action_text])
        observation, reward, done = process_ob(observation[0]), info["won"][0], done[0]
        trajectory.append(StepRecord(kind="ob", text=observation))
        prompt += f" {action_text}\n{observation}\n>"
        #print(f" {action_text}\n{observation}\n>")
        if done:
            return reward, True, trajectory
    return 0.0, False, trajectory


def parse_json_array(text: str) -> List[str]:
    try:
        start = text.find("[")
        end = text.rfind("]")
        arr = json.loads(text[start : end + 1]) if start >= 0 and end > start else []
        if isinstance(arr, list):
            return [str(x).strip() for x in arr if str(x).strip()]
    except Exception:
        pass
    return []


def bloom_failure_reasons(
    task_key: str,
    goal_ob: str,
    failed_trajectory_text: str,
    examples: str,
    successful_trajectory_text: str,
    trajectory_analysis: str,
    seed_reasons: List[str],
    client: OpenAI,
    model: str,
    max_tokens: int,
    k: int = 10,
) -> List[str]:
    seed_reasons = [s.strip() for s in seed_reasons if s and s.strip()]
    prompt = (
        "You are a failure-cause guess model for ALFWorld.\n"
        + ACTION_SPACE_INTRO
        + f"Task type: {task_key}\n"
        + f"Successful references:\n{examples}\n"
        #+ f"Successful trajectory from example-guided rerun:\n{successful_trajectory_text}\n"
        + f"Failed trajectory:\n{failed_trajectory_text}\n"
        + f"Analysis of successful vs failed trajectories:\n{trajectory_analysis}\n"
        + (f"Known useful historical causes for this task type: {seed_reasons}\n" if seed_reasons else "")
        + f"List {k} plausible failure causes as concise strings.You can think about the game's rule,the wrong action,the misunderstand of goal,and so on.Make sure that eash cause can guide the trajectory to success independently.Each cause more than 20 words.\n"
        + "Output JSON array only.\n"
    )
    out = llm2(prompt, client=client, model=model, max_tokens=max_tokens)
    reasons = parse_json_array(out)
    uniq: List[str] = []
    seen = set()
    for s in seed_reasons:
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
        if len(uniq) >= k:
            return uniq[:k]
    for r in reasons:
        key = r.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
        if len(uniq) >= k:
            break
    return uniq[:k]


def analyze_trajectory_differences(
    task_key: str,
    goal_ob: str,
    examples: str,
    successful_trajectory_text: str,
    failed_trajectory_text: str,
    client: OpenAI,
    model: str,
    max_tokens: int,
) -> str:
    prompt = (
        "You are an analysis robot for ALFWorld.\n"
        + ACTION_SPACE_INTRO
        + f"Task type: {task_key}\n"
        + f"Task goal/introduction:\n{goal_ob}\n"
        + f"Successful examples:\n{examples}\n"
        #+ f"Successful trajectory from example-guided rerun:\n{successful_trajectory_text}\n"
        + f"Failed trajectory:\n{failed_trajectory_text}\n\n"
        + "When summarizing trajectories, abstract them into short action segments and DO NOT mention concrete object names.\n"
        + "Examples:\n"
        + "[take one thing]->[put it in place]->[take another thing]->[put it in place]\n"
        + "[take one thing]->[cool it with fridge]->[put it in place]\n"
        + "Now you need to analyze briefly:\n"
        + "1) key behavior patterns in successful examples\n"
        + "2) key failure patterns in failed trajectory\n"
        + "3) main differences likely causing failure\n"
        + "Output concise analysis text only.\n"
    )
    return llm2(prompt, client=client, model=model, max_tokens=max_tokens).strip()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "base_config.yaml"), "r", encoding="utf-8") as reader:
        config = yaml.safe_load(reader)
    split = os.getenv("ALFWORLD_SPLIT", "eval_out_of_distribution")
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
    progress_path = os.path.join(base_dir, "bloom_dist_progress.json")
    bloom_memory = load_bloom_memory(memory_path)
    progress = load_progress(progress_path)
    start_idx = int(progress.get("next_idx", 0)) if progress else 0
    rs = progress.get("rs", [0] * 6) if progress else [0] * 6
    cnts = progress.get("cnts", [0] * 6) if progress else [0] * 6
    success_step_sums = progress.get("success_step_sums", [0] * 6) if progress else [0] * 6
    success_step_cnts = progress.get("success_step_cnts", [0] * 6) if progress else [0] * 6
    total_rerun_attempts = int(progress.get("total_rerun_attempts", 0)) if progress else 0
    total_success_reason_rank = int(progress.get("total_success_reason_rank", 0)) if progress else 0
    success_reason_count = int(progress.get("success_reason_count", 0)) if progress else 0
    if start_idx > 0:
        print(f"resume_from_idx={start_idx}")
        for idx in range(start_idx):
            env.reset()
    try:
        for idx in range(start_idx, 134):
            ob, info = env.reset()
            ob_text = "\n".join(ob[0].split("\n\n")[1:])
            name = "/".join(info["extra.gamefile"][0].split("/")[-3:-1])
            game_file = info["extra.gamefile"][0]
            print(f"task_start idx={idx + 1} name={name}")
            r = 0.0
            rerun_attempts = 0
            success_reason_rank = 0
            for i, (task_key, task_tag) in enumerate(PREFIXES.items()):
                if not name.startswith(task_key):
                    continue
                examples = build_same_type_examples(prompts, task_tag, top_n=2)
                init_prompt = f"Interact with a household to solve a task.Here is the task.\n" +ob_text + "\n>"
                #print(init_prompt)
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
                    print(trajectory_analysis)
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
                    print("bloom\n",reason_candidates)
                    for attempt_idx, reason in enumerate(reason_candidates, start=1):
                        rerun_attempts = attempt_idx
                        sim_env = create_single_game_env(game_file, config)
                        ob2, info2 = sim_env.reset()
                        ob2_text = "\n".join(ob2[0].split("\n\n")[1:])
                        retry_prompt = f"Interact with a household to solve a task.Here is the task.\n" +ob2_text + "\n>"
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
                            success_reason_rank = attempt_idx
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
            total_rerun_attempts += rerun_attempts
            if success_reason_rank > 0:
                total_success_reason_rank += success_reason_rank
                success_reason_count += 1
            avg_rerun_attempts = total_rerun_attempts / (idx + 1)
            avg_success_reason_rank = total_success_reason_rank / success_reason_count if success_reason_count > 0 else 0.0
            print(
                idx + 1,
                "r",
                r,
                "rerun_attempts",
                rerun_attempts,
                "avg_rerun_attempts",
                avg_rerun_attempts,
                "success_reason_rank",
                success_reason_rank,
                "avg_success_reason_rank",
                avg_success_reason_rank,
                "success_reason_count",
                success_reason_count,
                "rs",
                rs,
                "cnts",
                cnts,
                "sum(rs)/sum(cnts)",
                sum(rs) / max(1, sum(cnts))
            )
            save_progress(
                progress_path,
                {
                    "next_idx": idx + 1,
                    "rs": rs,
                    "cnts": cnts,
                    "success_step_sums": success_step_sums,
                    "success_step_cnts": success_step_cnts,
                    "total_rerun_attempts": total_rerun_attempts,
                    "total_success_reason_rank": total_success_reason_rank,
                    "success_reason_count": success_reason_count,
                    "memory_path": memory_path,
                    "k": k,
                },
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
                "total_rerun_attempts": total_rerun_attempts,
                "total_success_reason_rank": total_success_reason_rank,
                "success_reason_count": success_reason_count,
                "memory_path": memory_path,
                "k": k,
            },
        )
        raise
    avg_success_steps_by_type = [
        (success_step_sums[i] / success_step_cnts[i]) if success_step_cnts[i] > 0 else 0.0
        for i in range(6)
    ]
    total_success_steps = sum(success_step_sums)
    total_success_count = sum(success_step_cnts)
    overall_avg_success_steps = total_success_steps / total_success_count if total_success_count > 0 else 0.0
    category_names = list(PREFIXES.keys())
    result_path = os.path.join(base_dir, "alf_bloom_result.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"k={k}\n")
        f.write(f"success={rs}\n")
        f.write(f"count={cnts}\n")
        f.write(f"overall_success_rate={sum(rs) / max(1, sum(cnts))}\n")
        f.write(f"success_step_sums={success_step_sums}\n")
        f.write(f"success_step_counts={success_step_cnts}\n")
        f.write(f"avg_success_steps_by_type={avg_success_steps_by_type}\n")
        f.write(f"overall_avg_success_steps={overall_avg_success_steps}\n")
        for i, name in enumerate(category_names):
            f.write(
                f"{name}: success={rs[i]}, count={cnts[i]}, "
                f"avg_success_steps={avg_success_steps_by_type[i]}\n"
            )
        f.write(f"memory_file={memory_path}\n")
        f.write(f"progress_file={progress_path}\n")
    print("result_file", result_path)
    print("memory_file", memory_path)


if __name__ == "__main__":
    main()
