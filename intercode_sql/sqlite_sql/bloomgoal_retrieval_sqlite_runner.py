"""BloomGoal 算法 for InterCode SQLite.

对齐 codex_react_webshop/BloomGoal_webshop_runner.py 的全部功能：
- 独立 *_MAX_TOKENS 配置
- Memory 五字段（reflections / success_trajectories / goal_learning_notes / scoring_lessons / reflection_scoring_lessons + reflection_success_score_cases）
- annotated_trajectory 构建
- 三层决策环：候选生成 → 打分排序 → 最终选择（每一步都用独立 max_tokens）
- Reflection：生成 → 打分 → 选择（带 reflection_scores 加权）
- 学习：maybe_update_learning_notes（成功成功）/ record_reflection_success_case（失败先反思后成功）/ learn_reflection_scoring_from_successful_task
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests

from sqlite_sql.env import ACTION_EXEC, SQLiteSqlEnv
from sqlite_sql.runner_support import (
    REACT_SQL_DEMO,
    load_env_file,
    parse_react_action,
    strip_terminal_semicolon,
    truncate_observation,
)


DEFAULT_MEMORY_PATH = None
DEFAULT_SUCCESS_REWARD = float(os.environ.get("SUCCESS_REWARD", "1.0"))
MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
MAX_COMPLETION_TOKENS = max(16384, int(os.environ.get("OPENAI_MAX_TOKENS", "16384")))
USE_PROXY = os.environ.get("USE_PROXY", "0") == "1"

REFLECTION_SELECTION_MAX_TOKENS = max(MAX_COMPLETION_TOKENS, int(os.environ.get("BLOOMGOAL_REFLECTION_SELECTION_MAX_TOKENS", str(MAX_COMPLETION_TOKENS))))
FINAL_ACTION_MAX_TOKENS = max(MAX_COMPLETION_TOKENS, int(os.environ.get("BLOOMGOAL_FINAL_ACTION_MAX_TOKENS", str(MAX_COMPLETION_TOKENS))))
CANDIDATE_GENERATION_MAX_TOKENS = max(MAX_COMPLETION_TOKENS, int(os.environ.get("BLOOMGOAL_CANDIDATE_GENERATION_MAX_TOKENS", str(MAX_COMPLETION_TOKENS))))
ACTION_SCORING_MAX_TOKENS = max(MAX_COMPLETION_TOKENS, int(os.environ.get("BLOOMGOAL_ACTION_SCORING_MAX_TOKENS", str(MAX_COMPLETION_TOKENS))))
REFLECTION_GENERATION_MAX_TOKENS = max(MAX_COMPLETION_TOKENS, int(os.environ.get("BLOOMGOAL_REFLECTION_GENERATION_MAX_TOKENS", str(MAX_COMPLETION_TOKENS))))
LEARNING_NOTES_MAX_TOKENS = max(MAX_COMPLETION_TOKENS, int(os.environ.get("BLOOMGOAL_LEARNING_NOTES_MAX_TOKENS", str(MAX_COMPLETION_TOKENS))))
REFLECTION_SCORING_MAX_TOKENS = max(MAX_COMPLETION_TOKENS, int(os.environ.get("BLOOMGOAL_REFLECTION_SCORING_MAX_TOKENS", str(MAX_COMPLETION_TOKENS))))

MAX_REFLECTIONS_DEFAULT = int(os.environ.get("BLOOMGOAL_MAX_REFLECTIONS", "50"))

# 每任务 API 调用统计累加器（每个任务在独立进程内顺序执行，任务开始时清零，结束时快照）
_API_STATS = {
    "calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "latency_ms": 0.0,
}


def _reset_api_stats() -> None:
    _API_STATS["calls"] = 0
    _API_STATS["prompt_tokens"] = 0
    _API_STATS["completion_tokens"] = 0
    _API_STATS["total_tokens"] = 0
    _API_STATS["latency_ms"] = 0.0


def _snapshot_api_stats() -> Dict[str, Any]:
    return {
        "api_calls": _API_STATS["calls"],
        "prompt_tokens": _API_STATS["prompt_tokens"],
        "completion_tokens": _API_STATS["completion_tokens"],
        "total_tokens": _API_STATS["total_tokens"],
        "latency_ms": round(_API_STATS["latency_ms"], 1),
    }


def _loads_json_fragment(text: str) -> Any:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text or ""):
        if char not in "[{":
            continue
        try:
            data, _ = decoder.raw_decode(text[index:])
            return data
        except Exception:
            continue
    return None


def _format_action(action: str) -> str:
    action = (action or "").strip()
    if action.lower() == "submit":
        return "submit"
    return f"execute[{action}]"


def _normalize_candidate_action(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.lower() == "submit":
        return "submit"
    action, ok = parse_react_action(text)
    if ok:
        return action
    if text.lower().startswith("execute["):
        action, ok = parse_react_action("Action 1: " + text)
        if ok:
            return action
    return strip_terminal_semicolon(text)


def _dedupe_preserve_order(items: Iterable[Any], limit: int) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    effective_limit = limit if limit > 0 else len(list(items))
    for item in items:
        action = _normalize_candidate_action(item)
        if not action or action in seen:
            continue
        seen.add(action)
        out.append(action)
        if len(out) >= effective_limit:
            break
    return out


def parse_candidate_actions(raw: str, limit: int = 10) -> List[str]:
    data = _loads_json_fragment(raw)
    candidates: List[Any] = []
    if isinstance(data, list):
        candidates = data
    elif isinstance(data, dict):
        raw_actions = data.get("actions") or data.get("candidates") or data.get("candidate_actions")
        if isinstance(raw_actions, list):
            candidates = raw_actions
    if candidates:
        normalized_items: List[Any] = []
        for item in candidates:
            if isinstance(item, dict):
                item = item.get("action") or item.get("candidate") or item.get("sql") or item.get("text") or ""
            normalized_items.append(item)
        return _dedupe_preserve_order(normalized_items, limit)

    raw_items: List[Any] = []
    for line in (raw or "").splitlines():
        if "execute[" in line.lower():
            action, ok = parse_react_action(line)
            if ok:
                raw_items.append(action)
                continue
        if re.search(r"\bsubmit\b", line, flags=re.IGNORECASE):
            raw_items.append("submit")
    if not raw_items and raw:
        action, ok = parse_react_action(raw)
        if ok:
            raw_items.append(action)
        elif re.search(r"\bsubmit\b", raw, flags=re.IGNORECASE):
            raw_items.append("submit")
    return _dedupe_preserve_order(raw_items, limit)


def _clamp_distance(value: Any) -> float:
    try:
        distance = float(value)
    except Exception:
        distance = 1.0
    if distance < 0.0:
        return 0.0
    if distance > 1.0:
        return 1.0
    return distance


def parse_action_scores(raw: str, candidates: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    candidate_set = set(candidates)
    result: Dict[str, Dict[str, Any]] = {action: {"distance": 1.0, "rationale": ""} for action in candidates}
    data = _loads_json_fragment(raw)
    if isinstance(data, dict):
        maybe_scores = data.get("scores")
        if isinstance(maybe_scores, (list, dict)):
            data = maybe_scores
    if isinstance(data, dict):
        items = data.items()
    elif isinstance(data, list):
        items = []
        for item in data:
            if not isinstance(item, dict):
                continue
            action = item.get("action") or item.get("candidate") or item.get("sql") or ""
            items.append((action, item))
    else:
        items = []

    for action_raw, payload in items:
        action = _normalize_candidate_action(action_raw)
        if action not in candidate_set:
            continue
        if isinstance(payload, dict):
            result[action] = {
                "distance": _clamp_distance(
                    payload.get("distance", payload.get("goal_distance", payload.get("score", 1.0)))
                ),
                "rationale": str(payload.get("rationale", payload.get("reason", ""))).strip(),
                "advantages": payload.get("advantages", []),
                "risks": payload.get("risks", []),
                "rule_used": str(payload.get("rule_used", "")).strip(),
                "memory_effect": str(payload.get("memory_effect", "")).strip(),
            }
        else:
            result[action] = {"distance": _clamp_distance(payload), "rationale": ""}
    return result


def normalize_action_scores_zscore_minmax(scores: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """对 action 打分做标准化：先 Z-score 再 min-max，映射到 0-1。

    保留原始距离到 distance_raw 字段，distance 字段被替换为标准化后的值。
    """
    if not scores:
        return scores
    distances = [float(payload.get("distance", 1.0)) for payload in scores.values()]
    n = len(distances)
    mean = sum(distances) / n
    variance = sum((d - mean) ** 2 for d in distances) / n
    std = variance ** 0.5
    z_scores = {action: (float(payload.get("distance", 1.0)) - mean) / (std + 1e-8) for action, payload in scores.items()}
    z_values = list(z_scores.values())
    z_min = min(z_values)
    z_max = max(z_values)
    z_range = z_max - z_min
    normalized: Dict[str, Dict[str, Any]] = {}
    for action, payload in scores.items():
        raw = float(payload.get("distance", 1.0))
        z = z_scores[action]
        if z_range < 1e-8:
            norm = 0.5
        else:
            norm = (z - z_min) / z_range
        new_payload = dict(payload)
        new_payload["distance_raw"] = raw
        new_payload["distance"] = round(norm, 6)
        normalized[action] = new_payload
    return normalized


def rank_candidate_actions(candidates: Sequence[str], scores: Dict[str, Dict[str, Any]]) -> List[str]:
    indexed = list(enumerate(candidates))
    indexed.sort(key=lambda pair: (float(scores.get(pair[1], {}).get("distance", 1.0)), pair[0]))
    return [action for _, action in indexed]


def reflection_id(text: str) -> str:
    return "r_" + hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:12]


def normalize_reflection(item: Any) -> Optional[Dict[str, Any]]:
    if isinstance(item, str):
        text = item.strip()
    elif isinstance(item, dict):
        text = str(item.get("text") or item.get("reflection") or item.get("advice") or "").strip()
    else:
        text = ""
    if not text:
        return None
    return {"id": reflection_id(text), "text": text, "source": str(item.get("source", "")) if isinstance(item, dict) else ""}


def parse_reflections(raw: str, limit: int = 8) -> List[Dict[str, Any]]:
    data = _loads_json_fragment(raw)
    raw_items: List[Any] = []
    if isinstance(data, list):
        raw_items = data
    elif isinstance(data, dict):
        items = data.get("reflections") or data.get("reasons") or data.get("candidates")
        if isinstance(items, list):
            raw_items = items
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    if not raw_items:
        for line in (raw or "").splitlines():
            line = line.strip().lstrip("-").strip()
            if line:
                raw_items.append(line)
    for item in raw_items:
        normalized = normalize_reflection(item)
        if not normalized or normalized["id"] in seen:
            continue
        seen.add(normalized["id"])
        out.append(normalized)
        if len(out) >= limit:
            break
    return out


def parse_selected_reflection(
    raw: str, reflections: Sequence[Dict[str, Any]], tried_ids: Set[str]
) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, Any]]:
    by_id = {str(r.get("id", "")): r for r in reflections}
    data = _loads_json_fragment(raw)
    selected_id = ""
    if isinstance(data, dict):
        selected_id = str(
            data.get("selected_reflection_id") or data.get("reflection_id") or data.get("id") or ""
        ).strip()
        text = str(data.get("text") or data.get("reflection") or "").strip()
        if not selected_id and text:
            selected_id = reflection_id(text)
    elif isinstance(data, str):
        selected_id = data.strip()
    if selected_id in by_id and selected_id not in tried_ids:
        return by_id[selected_id], raw, {}
    for rid, r in by_id.items():
        if rid not in tried_ids:
            return r, raw, {}
    return None, raw, {}


def load_memory(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"reflections": [], "success_trajectories": [], "goal_learning_notes": "", "scoring_lessons": [], "reflection_scoring_lessons": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"reflections": [], "success_trajectories": [], "goal_learning_notes": "", "scoring_lessons": [], "reflection_scoring_lessons": []}
    if not isinstance(data, dict):
        return {"reflections": [], "success_trajectories": [], "goal_learning_notes": "", "scoring_lessons": [], "reflection_scoring_lessons": []}
    data.setdefault("reflections", [])
    data.setdefault("success_trajectories", [])
    data.setdefault("goal_learning_notes", "")
    data.setdefault("scoring_lessons", [])
    data.setdefault("reflection_success_score_cases", [])
    data.setdefault("reflection_scoring_lessons", [])
    return data


def save_memory(path: Path, memory: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_run_paths(
    out_dir: Optional[Path],
    log_path: Optional[Path],
    memory_path: Optional[Path],
) -> Tuple[Path, Path]:
    if out_dir is not None:
        result_dir = Path(out_dir)
    elif log_path is not None:
        result_dir = Path(log_path).parent
    elif memory_path is not None:
        result_dir = Path(memory_path).parent
    else:
        from uuid import uuid4
        result_dir = Path("results") / (
            time.strftime("bloomgoal_%Y%m%d_%H%M%S_") + uuid4().hex[:8]
        )
    result_dir.mkdir(parents=True, exist_ok=True)
    resolved_memory = (
        Path(memory_path)
        if memory_path is not None
        else result_dir / "BloomGoal_SQL_memory.json"
    )
    return result_dir, resolved_memory


def _next_seq(memory: Dict[str, Any]) -> int:
    seq = int(memory.get("_next_seq", 0)) + 1
    memory["_next_seq"] = seq
    return seq


def _reflection_priority(item: Dict[str, Any]) -> Tuple[int, int, int, int]:
    return (
        1 if item.get("confirmed") else 0,
        int(item.get("success_count", 0)),
        int(item.get("use_count", 0)),
        int(item.get("last_used_seq", item.get("created_seq", 0))),
    )


def upsert_reflection(
    memory: Dict[str, Any],
    text: str,
    session: str,
    confirmed: bool,
    success: bool,
    max_reflections: int = 50,
) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        raise ValueError("reflection text cannot be empty")
    rid = reflection_id(text)
    reflections = memory.setdefault("reflections", [])
    existing = None
    for item in reflections:
        if item.get("id") == rid:
            existing = item
            break
    if existing is None and not confirmed:
        return None
    seq = _next_seq(memory)
    if existing is None:
        existing = {
            "id": rid,
            "text": text,
            "confirmed": False,
            "success_count": 0,
            "failure_count": 0,
            "use_count": 0,
            "created_seq": seq,
            "last_used_seq": seq,
            "last_session": session,
        }
        reflections.append(existing)
    existing["text"] = text
    existing["confirmed"] = bool(existing.get("confirmed") or confirmed)
    existing["use_count"] = int(existing.get("use_count", 0)) + 1
    existing["last_used_seq"] = seq
    existing["last_session"] = session
    if success:
        existing["success_count"] = int(existing.get("success_count", 0)) + 1
    else:
        existing["failure_count"] = int(existing.get("failure_count", 0)) + 1
    reflections.sort(key=_reflection_priority, reverse=True)
    del reflections[max_reflections:]
    return existing


def is_unsafe_reflection_text(text: str) -> bool:
    lowered = (text or "").lower()
    unsafe_markers = [
        "ask user",
        "ask one",
        "clarifying question",
        "human only",
        "human-only",
    ]
    return any(marker in lowered for marker in unsafe_markers)


def memory_reflections(memory: Dict[str, Any], limit: int = 20) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in memory.get("reflections", []):
        reflection = normalize_reflection(
            {"id": item.get("id"), "text": item.get("text"), "source": "memory"}
        )
        if not reflection:
            continue
        reflection["confirmed"] = bool(item.get("confirmed"))
        if not reflection["confirmed"]:
            continue
        reflection["success_count"] = int(item.get("success_count", 0))
        reflection["failure_count"] = int(item.get("failure_count", 0))
        if is_unsafe_reflection_text(str(reflection.get("text", ""))):
            continue
        out.append(reflection)
        if len(out) >= limit:
            break
    return out


def compact_candidate_scores(scores: Any) -> Dict[str, Optional[float]]:
    compact: Dict[str, Optional[float]] = {}
    if not isinstance(scores, dict):
        return compact
    for action, score in scores.items():
        value = score.get("distance") if isinstance(score, dict) else score
        try:
            compact[str(action)] = float(value)
        except Exception:
            compact[str(action)] = None
    return compact


def build_annotated_success_trajectory(
    steps: Sequence[Dict[str, Any]], model_events: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    annotated: List[Dict[str, Any]] = []
    eventual_success = bool(steps and steps[-1].get("done") and float(steps[-1].get("reward") or 0.0) >= DEFAULT_SUCCESS_REWARD)
    for event in model_events:
        step_index = int(event.get("turn", event.get("step", -1)))
        if step_index < 0 or step_index >= len(steps):
            continue
        before = steps[step_index]
        after = steps[step_index + 1] if step_index + 1 < len(steps) else {}
        selected = str(event.get("normalized_action") or "")
        scores = event.get("scores") or {}
        selected_score = scores.get(selected, {}) if isinstance(scores, dict) else {}
        candidate_scores = compact_candidate_scores(scores)
        selected_distance = candidate_scores.get(selected)
        observation_before = str(before.get("observation", ""))
        observation_after = str(after.get("observation", ""))
        record: Dict[str, Any] = {
            "step": step_index,
            "observation_before": observation_before,
            "candidate_scores": candidate_scores,
            "selected_action": selected,
            "decision_type": event.get("decision_type", "execute"),
            "selected_distance": selected_distance,
            "selected_rationale": selected_score.get("rationale", "") if isinstance(selected_score, dict) else "",
        }
        if observation_after != observation_before:
            record["observation_after"] = observation_after
        reward_after = after.get("reward", before.get("reward", 0.0))
        try:
            has_reward = float(reward_after or 0.0) != 0.0
        except Exception:
            has_reward = bool(reward_after)
        if has_reward:
            record["reward_after"] = reward_after
            record["done_after"] = bool(after.get("done", False))
            record["eventual_success"] = eventual_success
        annotated.append(record)
    return annotated


def append_success_trajectory(
    memory: Dict[str, Any],
    session: str,
    reward: float,
    trajectory_text: str,
    annotated_trajectory: Optional[Sequence[Dict[str, Any]]] = None,
    max_successes: int = 30,
) -> None:
    successes = memory.setdefault("success_trajectories", [])
    successes.append(
        {
            "session": session,
            "reward": reward,
            "trajectory": trajectory_text,
            "annotated_trajectory": list(annotated_trajectory or []),
            "seq": _next_seq(memory),
        }
    )
    memory["success_trajectory_total"] = int(memory.get("success_trajectory_total", 0)) + 1
    del successes[:-max_successes]


def format_trajectory(query: str, history: Dict[str, List[Any]]) -> str:
    lines = [f"Question: {query}"]
    for idx, action in enumerate(history.get("actions", []), start=1):
        thought = history.get("thoughts", [""] * len(history["actions"]))[idx - 1]
        observation = history.get("observations", [""] * len(history["actions"]))[idx - 1]
        lines.append(str(thought))
        lines.append(f"Action {idx}: {_format_action(action)}")
        lines.append(f"Observation {idx}: {observation}")
    return "\n".join(lines)


def _similarity_terms(text: str) -> List[str]:
    lowered = str(text or "").lower()
    words = re.findall(r"[a-z0-9_]+", lowered)
    chinese = re.findall(r"[\u4e00-\u9fff]", lowered)
    return words + chinese + [
        chinese[index] + chinese[index + 1]
        for index in range(len(chinese) - 1)
    ]


def select_relevant_scoring_lessons(
    memory: Dict[str, Any],
    current_trajectory: str,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    lessons = [
        item for item in memory.get("scoring_lessons", [])
        if isinstance(item, dict)
        and str(item.get("scoring_lesson") or item.get("lesson") or "").strip()
    ]
    if limit <= 0 or not lessons:
        return []
    query_counts = Counter(_similarity_terms(current_trajectory))
    document_counts = [Counter(_similarity_terms(item.get("before_state", ""))) for item in lessons]
    document_frequency = Counter()
    for counts in document_counts:
        document_frequency.update(counts.keys())
    document_count = len(document_counts)

    def weighted(counts: Counter) -> Dict[str, float]:
        return {
            term: float(count) * (math.log((document_count + 1) / (document_frequency.get(term, 0) + 1)) + 1.0)
            for term, count in counts.items()
        }

    query_vector = weighted(query_counts)
    query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
    scored = []
    for index, (lesson, counts) in enumerate(zip(lessons, document_counts)):
        vector = weighted(counts)
        vector_norm = math.sqrt(sum(value * value for value in vector.values()))
        denominator = query_norm * vector_norm
        similarity = (
            sum(query_vector.get(term, 0.0) * value for term, value in vector.items()) / denominator
            if denominator else 0.0
        )
        scored.append((similarity, index, lesson))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [lesson for _, _, lesson in scored[:limit]]


def compact_memory_context(memory: Dict[str, Any], current_trajectory: str = "") -> str:
    lessons = []
    for item in select_relevant_scoring_lessons(memory, current_trajectory, limit=5):
        if isinstance(item, dict):
            lesson = str(item.get("scoring_lesson") or item.get("lesson") or "").strip()
            before_state = str(item.get("before_state") or "").strip()
            step = str(item.get("step") or "").strip()
            if lesson:
                prefix_parts = []
                if before_state:
                    prefix_parts.append(f"before={before_state}")
                if step:
                    prefix_parts.append(f"step={step}")
                prefix = " | ".join(prefix_parts)
                lessons.append(f"- {prefix}: {lesson}" if prefix else f"- {lesson}")
        else:
            lesson = str(item).strip()
            if lesson:
                lessons.append(f"- {lesson}")
    return "\n".join(
        [
            "Scoring lessons from underestimated successful actions:",
            "\n".join(lessons) if lessons else "None",
        ]
    )


def compact_reflection_score_context(memory: Dict[str, Any]) -> str:
    lessons = []
    for item in memory.get("reflection_scoring_lessons", [])[-10:]:
        if isinstance(item, dict):
            lesson = str(item.get("scoring_lesson") or item.get("lesson") or "").strip()
            direction = str(item.get("direction") or item.get("type") or "").strip()
            if lesson:
                tag = f"[{direction}] " if direction else ""
                lessons.append(f"- {tag}{lesson}")
        else:
            lesson = str(item).strip()
            if lesson:
                lessons.append(f"- {lesson}")
    return "\n".join(
        [
            "Reflection scoring lessons:",
            "\n".join(lessons) if lessons else "None",
        ]
    )


def llm_call(
    prompt: str,
    *,
    system: str,
    max_tokens: int = MAX_COMPLETION_TOKENS,
    temperature: float = 0.0,
    retries: int = 6,
) -> Tuple[str, Dict[str, Any]]:
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY is not set on remote")
    headers = {
        "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
        "Content-Type": "application/json",
    }
    proxies = (
        {"http": "http://127.0.0.1:7897", "https": "http://127.0.0.1:7897"}
        if USE_PROXY
        else None
    )
    payload_base = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    last_error: Optional[BaseException] = None
    call_start = time.perf_counter()
    for attempt in range(1, retries + 1):
        payload = dict(payload_base)
        payload["max_completion_tokens" if attempt <= 3 else "max_tokens"] = max_tokens
        try:
            _API_STATS["calls"] += 1
            response = requests.post(
                BASE_URL + ("/chat/completions" if BASE_URL.endswith("/v1") else "/v1/chat/completions"),
                headers=headers,
                json=payload,
                timeout=240,
                proxies=proxies,
            )
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:800]}")
            data = response.json()
            text = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
            if text:
                usage = data.get("usage", {}) or {}
                _API_STATS["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                _API_STATS["completion_tokens"] += int(usage.get("completion_tokens") or 0)
                _API_STATS["total_tokens"] += int(usage.get("total_tokens") or 0)
                _API_STATS["latency_ms"] += (time.perf_counter() - call_start) * 1000
                return text, usage
            last_error = RuntimeError("empty model content: " + json.dumps(data.get("usage", {})))
        except Exception as exc:
            last_error = exc
        time.sleep(min(30, 2 * attempt))
    raise RuntimeError(f"LLM call failed after {retries} retries: {last_error}")


def schema_hint_from_record(record: Dict[str, Any]) -> str:
    return "Schema is initially unknown. Use SHOW TABLES and DESC table to discover tables and columns."


# ---------------------------------------------------------------------------
# PROMPTS (领域特定：SQLite InterCode)
# ---------------------------------------------------------------------------

# FAILURE_REFLECTION_FEW_SHOTS = """Failure reflection few-shots:
# [
#   {"text": "First run SHOW TABLES, then DESC suspected tables before drafting a query that references unknown columns or joins.", "source": "Planner Critic"},
#   {"text": "If the SQL result set is empty, verify the literal value matches the column values exactly (case sensitive? extra spaces?).", "source": "Environment Critic"}
# ]
# """

REFLECTION_SCORING_FEW_SHOTS = """Reflection scoring few-shot:
Input pool:
- id=r_xxx confirmed=no: "Be more careful next time."
- id=r_yyy confirmed=no: "Run SHOW TABLES first if schema was unknown, then DESCRIBE each table before joins."
Output:
{"r_xxx": {"score": 0.1, "relevance": 0.1, "actionability": 0.05, "conciseness": 0.8, "rationale": "Vague advice, no concrete step."},
 "r_yyy": {"score": 0.92, "relevance": 0.9, "actionability": 0.95, "conciseness": 0.9, "rationale": "Specific, directly executable inspection sequence."}}
"""

SUCCESS_LEARNING_FEW_SHOTS = """Success scoring-learning few-shot (underestimated direction):
Input annotated trajectory step: last_observation="Error: no such column X", selected_action="execute[SELECT * FROM t]", selected_distance=0.1, selected_rationale="Looks good enough".
Output lesson:
[{"before_state": "SQL executor reported unknown column error", "step": "execute[SELECT * FROM t] with assumed existing columns", "scoring_lesson": "A query whose columns/table names are unverified after an error should receive distance >=0.7 unless DESCRIBE/SHOW TABLES was the immediately preceding step."}]
"""


def candidate_generation_prompt(
    *,
    query: str,
    schema_hint: str,
    recent_trace: str,
    latest_observation: Any,
    memory_context: str,
    active_reflection: str,
    max_candidates: int,
    failed_trajectories: Sequence[str] = (),
) -> str:
    trajectory_block = "\n\n".join(
        f"Failed same-task trajectory {i + 1}:\n{t}"
        for i, t in enumerate(failed_trajectories)
    ) or "(none)"
    return f"""# Role
You are generating candidate actions for an InterCode SQL ReAct agent.
The agent must answer the question by interacting with a SQL database.

# Allowed actions
- execute[SQL]
- submit

# Behavior pattern
Use this ReAct SQL few-shot as the behavior pattern:
{REACT_SQL_DEMO.strip()}

# Current question
{query}
{schema_hint}

# Recent trajectory
{recent_trace or "(empty)"}

# Latest observation
{truncate_observation(latest_observation)}

# Memory (advisory only)
{memory_context or "(none)"}

# Failed same-task trajectory
{trajectory_block}

# reflection for failed same-task trajectory
{active_reflection or "(none)"}

# Output format
Return JSON only with an "actions" array of at most {max_candidates} candidate actions.
Prefer legal, useful SQL inspection or answer actions. Include submit only if the latest result is sufficient.
"""


def action_scoring_prompt(
    *,
    candidates: Sequence[str],
    query: str,
    schema_hint: str,
    recent_trace: str,
    latest_observation: Any,
    memory_context: str,
    active_reflection: str,
    failed_trajectories: Sequence[str] = (),
) -> str:
    formatted = "\n".join(f"- {_format_action(action)}" for action in candidates)
    trajectory_block = "\n\n".join(
        f"Failed same-task trajectory {i + 1}:\n{t}"
        for i, t in enumerate(failed_trajectories)
    ) or "(none)"
    return f"""# Role
You are the Action Scorer for InterCode SQL.
Score each candidate by goal distance from 0.0 to 1.0 where lower is closer to answering the current SQL question.

# Primary objective
Solve the current question using the current database evidence.
Memory and Active reflection are advisory references; ignore them when they conflict with the question, SQL legality, or the latest observation.

# Important scoring rules
- Useful schema inspection (SHOW TABLES, DESC table, PRAGMA table_info) is low-distance when table or column names are uncertain.
- A SQL query that directly answers the question is low-distance when its tables and columns are supported by prior inspection or observation.
- Repeating a failed or irrelevant query is high-distance.
- submit is low-distance only when the latest SQL result answers the question; otherwise submit is high-distance.
- Do not treat matching memory or reflection as a standalone reward objective. Instead, task completion serves as the ultimate optimization target, while memory and reflection act only as auxiliary references.

# Behavior pattern
ReAct SQL few-shot:
{REACT_SQL_DEMO.strip()}

# Current question
{query}
{schema_hint}

# Recent trajectory
{recent_trace or "(empty)"}

# Latest observation
{truncate_observation(latest_observation)}

# Memory
{memory_context or "(none)"}

# Failed same-task trajectory
{trajectory_block}

# reflection for failed same-task trajectory
{active_reflection or "(none)"}

# Candidates
{formatted}

# Output format
Return JSON only: {{"scores": [{{"action": "execute[...]", "distance": 0.0, "rationale": "...", "rule_used": "", "memory_effect": ""}}]}}
"""


def final_action_prompt(
    *,
    ranked_actions: Sequence[str],
    scores: Dict[str, Dict[str, Any]],
    query: str,
    recent_trace: str,
    latest_observation: Any,
    active_reflection: str,
    failed_trajectories: Sequence[str] = (),
) -> str:
    lines = []
    for action in ranked_actions:
        score = scores.get(action, {})
        lines.append(f"- {_format_action(action)} distance={score.get('distance', 1.0)} rationale={score.get('rationale', '')}")
    trajectory_block = "\n\n".join(
        f"Failed same-task trajectory {i + 1}:\n{t}"
        for i, t in enumerate(failed_trajectories)
    ) or "(none)"
    return f"""# Task
Choose the single next action for this InterCode SQL task.

# Output format
Return JSON only with "action" and optional "decision_type" / "rationale".

# Question
{query}

# Recent trajectory
{recent_trace or "(empty)"}

# Latest observation
{truncate_observation(latest_observation)}

# Failed same-task trajectory
{trajectory_block}

# reflection for failed same-task trajectory
{active_reflection or "(none)"}

# Ranked candidates
{chr(10).join(lines)}
"""


def failure_reflection_prompt(
    failed_trajectories: Sequence[str],
    memory_context: str,
    k: int,
    reflection_score_context: str = "",
) -> str:
    failure_blocks = "\n\n".join(
        f"Failed same-task trajectory {i + 1}:\n{trajectory}"
        for i, trajectory in enumerate(failed_trajectories)
    )
    scoring_hint = ""
    if reflection_score_context and "None" not in reflection_score_context:
        scoring_hint = (
            "# Scoring guidelines for high-quality reflections:\n"
            f"{reflection_score_context}\n\n"
            "# Follow these rules to maximize the scores your reflections will receive. "
            "High-scoring reflections are: highly relevant to the actual failure patterns in the trajectories, "
            "concrete and actionable for a rerun, and concise/professional in wording.\n"
            "Low-scoring reflections to avoid: vague generic advice, irrelevant to the actual failure, "
            "overly abstract suggestions not executable in practice.\n\n"
        )
    return f"""# Role
Generate failure reasons for an InterCode SQL task.

# Context
The Failures section contains previous failed trajectories from the same SQL task.
reward < 1 indicates the executed SQL did not produce the correct answer and results in task failure.
You must guess plausible failure causes from multiple angles and propose at least {k} distinct reasons.
Avoid duplicating failure reasons already recorded in Memory.

# Critic roles
- Planner Critic: whether the SQL decomposition / schema-inspection order was wrong.
- Action Critic: whether selected actions caused drift, repeated a bad query, ignored schema errors, or produced empty result sets.
- Environment Critic: whether column names, types, nulls, case sensitivity, join keys, or ordering/aggregation semantics were misunderstood.
- Demand Critic: whether the agent overinterpreted, omitted, or misread question requirements (e.g. LIMIT, DISTINCT, HAVING).
- Arbiter: merge the strongest diagnoses into actionable failure reasons.

# Output requirement
Each reason must be a concise strategy that could independently guide a rerun.
Return JSON array only, at least {k} objects, with fields "text" and "source".
"source" must come from the critic roles above; "text" is the reflection body, including the failure-trajectory description and problem attribution.

{scoring_hint}

# Correct demo pattern
ReAct SQL correct demo pattern:
{REACT_SQL_DEMO.strip()}

# Failures
{failure_blocks[-16000:]}

"""


def _format_reflection_lines(
    reflections: Sequence[Dict[str, Any]],
    tried_ids: Set[str],
    reflection_scores: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    lines = []
    for r in reflections:
        rid = str(r.get("id", ""))
        text = str(r.get("text", ""))
        confirmed = "yes" if r.get("confirmed") else "no"
        status = "tried" if rid in tried_ids else "available"
        score_info = ""
        if reflection_scores and rid in reflection_scores:
            s = reflection_scores[rid]
            score_info = f" score={s.get('score', -1):.2f} rel={s.get('relevance', -1):.2f} act={s.get('actionability', -1):.2f} con={s.get('conciseness', -1):.2f}"
        lines.append(f"- id={rid} confirmed={confirmed} status={status}{score_info}: {text}")
    return "\n".join(lines) if lines else "None"


def selection_prompt(
    verified_reflections: Sequence[Dict[str, Any]],
    new_reflections: Sequence[Dict[str, Any]],
    failed_trajectories: Sequence[str],
    tried_ids: Set[str],
    reflection_scores: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    pool = list(verified_reflections) + list(new_reflections)
    reflection_block = _format_reflection_lines(pool, tried_ids, reflection_scores)
    failure_blocks = "\n\n".join(
        f"Failed same-task trajectory {i + 1}:\n{t}" for i, t in enumerate(failed_trajectories[-5:])
    )
    verified_ids = {str(r.get("id", "")) for r in verified_reflections}
    new_ids = {str(r.get("id", "")) for r in new_reflections}
    return f"""# Task
Select one InterCode-SQL reflection for the next rerun.

# Selection rules
- Prefer reflections with a higher score when they are available, not tried, and relevant to the most recent failure.
- If verified (confirmed=yes) reflections exist and are not tried, strongly prefer them when their score or wording is comparable to new ones.
- Pick an available (not tried) reflection. Do not repeat already tried ones.
- If reflection scores exist, they have been computed from relevance / actionability / conciseness against the actual failed trajectories. Higher overall score means higher quality.

# Pool classification by source
- Verified (memory confirmed): {sorted(verified_ids)}
- New (generated this task): {sorted(new_ids)}

# Correct demo pattern
ReAct SQL correct demo pattern:
{REACT_SQL_DEMO.strip()}

# Reflection pool
{reflection_block}

# Recent failures
{failure_blocks[-10000:]}

# Output format
Return JSON only with: {{"selected_reflection_id": "r_..."}}
"""


def score_reflections_prompt(
    reflections: Sequence[Dict[str, Any]],
    failed_trajectories: Sequence[str],
    score_context: str,
) -> str:
    reflection_lines = []
    for r in reflections:
        rid = str(r.get("id", ""))
        text = str(r.get("text", ""))
        confirmed = "yes" if r.get("confirmed") else "no"
        reflection_lines.append(f"- id={rid} confirmed={confirmed}: {text}")
    reflection_block = "\n".join(reflection_lines) if reflection_lines else "None"
    failure_blocks = "\n\n".join(
        f"Failed trajectory {i + 1}:\n{trajectory}"
        for i, trajectory in enumerate(failed_trajectories[-5:])
    )
    return f"""# Role
You are the InterCode SQL trajectory Reflection Scorer.

# Scoring dimensions
Score each reflection on three dimensions (each 0-1):
1. relevance: How well the reflection matches the specific failure patterns in the failed trajectories.
2. actionability: How useful the advice is for guiding a successful rerun — concrete, specific, executable SQL/Schema steps.
3. conciseness: How direct, professional, and well-structured the wording is.

Overall score = average of the three dimensions, rounded to two decimals.

# Scoring notes
Reflections that are vague (e.g. "be more careful"), irrelevant to the actual failure, or too generic should receive low scores. Reflections that pinpoint the exact SQL error (schema unknown, missing aggregation, case mismatch, join keys) and give a concrete alternative workflow should receive high scores.

# Few-shots
{REFLECTION_SCORING_FEW_SHOTS}

# Correct demo
a correct ReAct SQL demo:
{REACT_SQL_DEMO.strip()}

# Scoring experience
{score_context}

# Reflections to score
{reflection_block}

# Failed trajectories
{failure_blocks}

# Output format
Output JSON object only. Keys must be reflection ids. Each value must be:
{{"score": number_0_to_1, "relevance": number_0_to_1, "actionability": number_0_to_1, "conciseness": number_0_to_1, "rationale": "brief reason for the score"}}.
"""


def parse_reflection_scores(
    raw: str, reflections: Sequence[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    reflection_ids = {str(r.get("id", "")) for r in reflections}
    data = _loads_json_fragment(raw)
    result: Dict[str, Dict[str, Any]] = {}
    if not isinstance(data, dict):
        return result
    for rid, payload in data.items():
        rid = str(rid).strip()
        if rid not in reflection_ids:
            continue
        if isinstance(payload, dict):
            score = float(payload.get("score", payload.get("overall", 0.5)))
            relevance = float(payload.get("relevance", 0.5))
            actionability = float(payload.get("actionability", 0.5))
            conciseness = float(payload.get("conciseness", 0.5))
            rationale = str(payload.get("rationale", "")).strip()
            result[rid] = {
                "score": max(0.0, min(1.0, score)),
                "relevance": max(0.0, min(1.0, relevance)),
                "actionability": max(0.0, min(1.0, actionability)),
                "conciseness": max(0.0, min(1.0, conciseness)),
                "rationale": rationale,
            }
        else:
            score = float(payload)
            result[rid] = {"score": max(0.0, min(1.0, score)), "relevance": score, "actionability": score, "conciseness": score, "rationale": ""}
    return result


def rank_reflections_by_score(
    reflections: Sequence[Dict[str, Any]],
    scores: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    indexed = list(reflections)
    indexed.sort(key=lambda r: -float(scores.get(str(r.get("id", "")), {}).get("score", 0.0)))
    return indexed


def score_reflections(
    reflections: Sequence[Dict[str, Any]],
    failed_trajectories: Sequence[str],
    memory: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], str, Dict[str, Any]]:
    if not reflections:
        return {}, "", {}
    score_context = compact_reflection_score_context(memory)
    raw, usage = llm_call(
        score_reflections_prompt(reflections, failed_trajectories, score_context),
        system="You score trajectory reflections for InterCode SQL failure recovery and output JSON only.",
        max_tokens=REFLECTION_SCORING_MAX_TOKENS,
    )
    return parse_reflection_scores(raw, reflections), raw, usage


def reflection_scoring_lesson_id(lesson: Dict[str, Any]) -> str:
    basis = "\n".join(
        [
            str(lesson.get("direction", "")).strip().lower(),
            str(lesson.get("scoring_lesson", "")).strip().lower(),
        ]
    )
    digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]
    return f"rsl_{digest}"


def _build_reflection_retry_sequence(
    retry_records: Sequence[Dict[str, Any]],
    successful_trajectory: str,
    all_scored_reflections: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    parts: List[str] = []
    if all_scored_reflections:
        scored_lines = []
        for rid, info in all_scored_reflections.items():
            s = float(info.get("score", -1))
            scored_lines.append(
                f"- id={rid} score={s:.2f} (rel={float(info.get('relevance', 0)):.2f} "
                f"act={float(info.get('actionability', 0)):.2f} con={float(info.get('conciseness', 0)):.2f}): "
                f"{info.get('text', '')}"
                + (f" | rationale: {info.get('rationale', '')}" if info.get("rationale") else "")
            )
        parts.append(
            "# Full reflection pool with scores\n"
            + ("\n".join(scored_lines) if scored_lines else "None")
        )
    retry_parts = []
    for i, rec in enumerate(retry_records):
        failed_traj = str(rec.get("failed_trajectory", ""))
        reflection_text = str(rec.get("reflection_text", ""))
        score_val = float(rec.get("score", 0))
        score_info = f"score={score_val:.2f}"
        score_rationale = str(rec.get("rationale", ""))
        dims = f"relevance={float(rec.get('relevance', 0)):.2f} actionability={float(rec.get('actionability', 0)):.2f} conciseness={float(rec.get('conciseness', 0)):.2f}"
        retry_parts.append(
            f"# Failed trajectory {i + 1}\n{failed_traj}\n"
            f"# Reflection used {score_info} ({dims})\n{reflection_text}\n"
            + (f"# Scoring rationale\n{score_rationale}\n" if score_rationale else "")
        )
    retry_parts.append(f"# Successful trajectory\n{successful_trajectory}")
    parts.append("# Retry sequence (reflections actually selected and executed)\n" + "\n---\n".join(retry_parts))
    return "\n\n---\n\n".join(parts)


def reflection_score_success_prompt(
    retry_records: Sequence[Dict[str, Any]],
    successful_trajectory: str,
    all_scored_reflections: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    sequence = _build_reflection_retry_sequence(retry_records, successful_trajectory, all_scored_reflections)
    return f"""# Role
You are the InterCode SQL reflection-score learner (underestimated direction).

# Context
The "Full reflection pool" section shows ALL reflections generated for this task with their scores. The "Retry sequence" shows which ones were actually selected and whether they led to success.

# Analysis target
Analyze: which reflections in the full pool were UNDERESTIMATED by the scorer?

# Definition: underestimated reflection
An underestimated reflection is one that:
- Received a LOW score (e.g. <0.7) in the pool, BUT
- Either it was selected and contributed to success, OR
- Its content actually aligns with the successful trajectory pattern, even if not selected.

# Focus points
- Pool reflections that received low relevance/actionability scores but contain insights that match the successful trajectory.
- Reflections that were concise but their depth was missed by the scorer.
- What reflection characteristics should receive HIGHER scores in the future?

# Output constraints
Do NOT output task-specific wording. Give reusable, abstract scoring rules. Output at most 1 lesson — only extract a genuinely novel and useful insight. If nothing is worth recording, return an empty array.

# Output format
Output JSON array only (0-1 items). Each item must have:
- "direction": "underestimated"
- "scoring_lesson": concise rule for future reflection scoring

# Example (1 item)
[{{"direction": "underestimated", "scoring_lesson": "Reflections that recommend SHOW TABLES + DESCRIBE before drafting queries on unknown schema deserve +0.2 actionability."}}]

# Example (empty)
[]

# Input data
{sequence}

# Hard limit
Output at most 1 lesson per call (empty array if nothing is worth recording).
"""


def reflection_score_failure_prompt(
    retry_records: Sequence[Dict[str, Any]],
    successful_trajectory: str,
    all_scored_reflections: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    sequence = _build_reflection_retry_sequence(retry_records, successful_trajectory, all_scored_reflections)
    return f"""# Role
You are the InterCode SQL reflection-score learner (overrate direction).

# Analysis target
Analyze: which reflections in the full pool were OVERRATED by the scorer?

# Definition: overrated reflection
An overrated reflection is one that:
- Received a HIGH score (e.g. >=0.75) in the pool, BUT
- It was selected and the retry still FAILED, OR
- Its content looks plausible but is actually vague, schema-incompatible, or practically unhelpful.

# Focus points
- Pool reflections that scored high on relevance/actionability but the advice was not executable in practice on a SQL engine.
- What reflection characteristics should receive LOWER scores in the future?

# Output constraints
Do NOT output task-specific wording. Give reusable, abstract scoring rules. Output at most 1 lesson. If nothing is worth recording, return empty array.

# Output format
Output JSON array only (0-1 items). Each item must have:
- "direction": "overrated"
- "scoring_lesson": concise rule for future reflection scoring

# Example (1 item)
[{{"direction": "overrated", "scoring_lesson": "Reflections that abstractly suggest 'check data more carefully' without mapping to a specific SQL inspection step should receive <=0.5 actionability."}}]

# Example (empty)
[]

# Input data
{sequence}

# Hard limit
Output at most 1 lesson per call (empty array if nothing is worth recording).
"""


def parse_reflection_score_lessons(raw: str) -> List[Dict[str, Any]]:
    data = _loads_json_fragment(raw)
    lessons: List[Dict[str, Any]] = []
    items: List[Any] = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for key in ("lessons", "scoring_lessons", "items"):
            if isinstance(data.get(key), list):
                items = data[key]
                break
        else:
            items = [data]
    for item in items:
        if not isinstance(item, dict):
            continue
        direction = str(item.get("direction") or item.get("type") or "").strip()
        scoring_lesson = str(item.get("scoring_lesson") or item.get("lesson") or "").strip()
        if not scoring_lesson:
            continue
        lessons.append({"direction": direction, "scoring_lesson": scoring_lesson})
    return lessons[:1]


def append_reflection_scoring_lessons(
    memory: Dict[str, Any],
    lessons: Sequence[Dict[str, Any]],
    max_lessons: int = 10,
) -> None:
    # max_lessons remains in the signature for caller compatibility; lessons are
    # intentionally retained for the full lifetime of this run's memory file.
    stored = memory.setdefault("reflection_scoring_lessons", [])
    seen = {str(item.get("id") or reflection_scoring_lesson_id(item)) for item in stored if isinstance(item, dict)}
    added = 0
    for lesson in lessons:
        if added >= 1:
            break
        if not isinstance(lesson, dict):
            continue
        scoring_lesson = str(lesson.get("scoring_lesson") or "").strip()
        if not scoring_lesson:
            continue
        direction = str(lesson.get("direction") or "").strip()
        normalized = {"direction": direction, "scoring_lesson": scoring_lesson}
        normalized["id"] = str(lesson.get("id") or reflection_scoring_lesson_id(normalized))
        if normalized["id"] in seen:
            continue
        seen.add(normalized["id"])
        stored.append(normalized)
        added += 1


def learn_reflection_scoring_from_successful_task(
    memory: Dict[str, Any],
    *,
    retry_records: Sequence[Dict[str, Any]],
    successful_trajectory: str,
    all_scored_reflections: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    if not retry_records:
        return
    raw_success, _ = llm_call(
        reflection_score_success_prompt(retry_records, successful_trajectory, all_scored_reflections),
        system="You derive scoring lessons from reflection sequences where low-scored reflections contributed to eventual success. Output JSON only.",
        max_tokens=LEARNING_NOTES_MAX_TOKENS,
    )
    append_reflection_scoring_lessons(memory, parse_reflection_score_lessons(raw_success))
    if len(retry_records) >= 2:
        raw_failure, _ = llm_call(
            reflection_score_failure_prompt(retry_records, successful_trajectory, all_scored_reflections),
            system="You derive scoring corrections from reflection sequences where high-scored reflections still led to failure. Output JSON only.",
            max_tokens=LEARNING_NOTES_MAX_TOKENS,
        )
        append_reflection_scoring_lessons(memory, parse_reflection_score_lessons(raw_failure))


def generate_failure_reflections(
    failed_trajectories: Sequence[str],
    memory_context: str,
    k: int,
    reflection_score_context: str = "",
) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    raw, usage = llm_call(
        failure_reflection_prompt(failed_trajectories, memory_context, k, reflection_score_context),
        system="You are a multi-critic reflection generator for failed InterCode SQL trajectories.",
        max_tokens=REFLECTION_GENERATION_MAX_TOKENS,
    )
    return parse_reflections(raw, limit=max(50, k * 3)), raw, usage


def select_best_reflection(
    verified_reflections: Sequence[Dict[str, Any]],
    new_reflections: Sequence[Dict[str, Any]],
    failed_trajectories: Sequence[str],
    tried_ids: Set[str],
    reflection_scores: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, Any]]:
    all_reflections = list(verified_reflections) + list(new_reflections)
    raw, usage = llm_call(
        selection_prompt(verified_reflections, new_reflections, failed_trajectories, tried_ids, reflection_scores),
        system="You select one InterCode SQL reflection for the next rerun. Output JSON only.",
        max_tokens=REFLECTION_SELECTION_MAX_TOKENS,
    )
    selection, raw_selection, usage_selection = parse_selected_reflection(raw, all_reflections, tried_ids)
    return selection, raw_selection, usage_selection


def success_learning_prompt(successes: Sequence[Dict[str, Any]]) -> str:
    payload = []
    for item in successes:
        compact_steps = []
        for step in item.get("annotated_trajectory", []):
            if not isinstance(step, dict):
                continue
            compact = {
                "step": step.get("step"),
                "last_observation": str(step.get("observation_before", "")),
                "selected_action": step.get("selected_action"),
                "selected_distance": step.get("selected_distance"),
                "selected_rationale": step.get("selected_rationale", ""),
            }
            for key in ("reward_after", "done_after", "eventual_success"):
                if key in step:
                    compact[key] = step.get(key)
            compact_steps.append(compact)
        payload.append(
            {
                "session": item.get("session"),
                "reward": item.get("reward"),
                "annotated_trajectory": compact_steps,
            }
        )
    return f"""# Role
You are the InterCode SQL action-score learner.

# Input description
You are given complete successful InterCode SQL trajectories with compact scored step annotations.
Each step includes last_observation, selected_action, selected_distance, and selected_rationale. The final non-zero-reward step may include reward_after, done_after, and eventual_success.

# Lesson extraction rule
Only extract lessons from underestimated successful actions: actions with high selected_distance in successful trajectories.

# Output constraints
Do NOT output question-specific wording, table names, literal values, or task IDs.
Return concise reusable scoring experience only.

# Output fields (per lesson)
1. before_state: abstract state before the underestimated action
2. step: which step/action pattern was underestimated, stated abstractly
3. scoring_lesson: the scoring experience lesson that should guide future Action Scorer distance estimates.

# Output format
Output JSON array only. Each item must have exactly these fields:
[
  {{
    "before_state": "abstract state before that step",
    "step": "which underestimated action step/pattern",
    "scoring_lesson": "concise scoring lesson"
  }}
]

# Few-shots
{SUCCESS_LEARNING_FEW_SHOTS}

# Complete scored successful trajectories
{json.dumps(payload, ensure_ascii=False)}

# Hard limit
Output at most 1 lesson per call (empty array if nothing is worth recording).
"""


def _sanitize_lesson_text(value: Any, limit: int = 700) -> str:
    text = re.sub(r"\b[A-Z_][A-Z0-9_]{2,}\b\s*\([^)]*\)", "query_shape", str(value or ""), flags=re.IGNORECASE)
    text = re.sub(r"\btable_[a-zA-Z0-9_]+\b", "unknown_table", text)
    text = re.sub(r"\bcol_[a-zA-Z0-9_]+\b", "unknown_column", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def parse_scoring_lessons(raw: str) -> List[Dict[str, Any]]:
    data = _loads_json_fragment(raw)
    if isinstance(data, dict):
        for key in ("scoring_lessons", "lessons", "items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data]
    lessons: List[Dict[str, Any]] = []
    if not isinstance(data, list):
        return lessons
    for item in data:
        if not isinstance(item, dict):
            continue
        lesson = _sanitize_lesson_text(item.get("scoring_lesson") or item.get("lesson") or item.get("how_future_scoring_should_adjust"), 700)
        if not lesson:
            continue
        before_state = _sanitize_lesson_text(item.get("before_state") or item.get("state_before") or item.get("state_pattern"), 500)
        step = _sanitize_lesson_text(item.get("step") or item.get("action_step") or item.get("underestimated_action_pattern"), 300)
        lessons.append({"before_state": before_state, "step": step, "scoring_lesson": lesson})
    return lessons[:1]


def scoring_lesson_id(lesson: Dict[str, Any]) -> str:
    basis = "\n".join(
        [
            str(lesson.get("before_state", "")).strip().lower(),
            str(lesson.get("step", "")).strip().lower(),
            str(lesson.get("scoring_lesson", "")).strip().lower(),
        ]
    )
    digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]
    return f"l_{digest}"


def append_scoring_lessons(memory: Dict[str, Any], lessons: Sequence[Dict[str, Any]], max_lessons: int = 10) -> None:
    # max_lessons remains in the signature for caller compatibility; retrieval
    # controls prompt size without deleting historical lessons.
    stored = memory.setdefault("scoring_lessons", [])
    seen = {str(item.get("id") or scoring_lesson_id(item)) for item in stored if isinstance(item, dict)}
    added = 0
    for lesson in lessons:
        if added >= 1:
            break
        if not isinstance(lesson, dict):
            continue
        normalized = {
            "before_state": _sanitize_lesson_text(lesson.get("before_state"), 500),
            "step": _sanitize_lesson_text(lesson.get("step"), 300),
            "scoring_lesson": _sanitize_lesson_text(lesson.get("scoring_lesson"), 700),
        }
        if not normalized["scoring_lesson"]:
            continue
        normalized["id"] = str(lesson.get("id") or scoring_lesson_id(normalized))
        if normalized["id"] in seen:
            continue
        seen.add(normalized["id"])
        stored.append(normalized)
        added += 1


def maybe_update_learning_notes(memory: Dict[str, Any], learn_every: int) -> None:
    if learn_every <= 0:
        return
    total = int(memory.get("success_trajectory_total", 0))
    if total < learn_every or total % learn_every != 0:
        return
    successes = [s for s in memory.get("success_trajectories", []) if s.get("annotated_trajectory")]
    recent = successes[-learn_every:]
    prompt = success_learning_prompt(recent)
    raw, _ = llm_call(
        prompt,
        system="You derive concise reusable scoring lessons from underestimated successful InterCode SQL actions.",
        max_tokens=LEARNING_NOTES_MAX_TOKENS,
    )
    append_scoring_lessons(memory, parse_scoring_lessons(raw))


def _compact_scored_attempt_for_failure_lesson(attempt: Dict[str, Any], max_steps: int = 80) -> Dict[str, Any]:
    compact_steps = []
    for step in list(attempt.get("annotated_trajectory") or [])[-max_steps:]:
        if not isinstance(step, dict):
            continue
        compact = {
            "step": step.get("step"),
            "before_state": str(step.get("observation_before", "")),
            "selected_action": step.get("selected_action"),
            "selected_distance": step.get("selected_distance"),
            "selected_rationale": step.get("selected_rationale", ""),
        }
        for key in ("reward_after", "done_after", "eventual_success"):
            if key in step:
                compact[key] = step.get(key)
        compact_steps.append(compact)
    return {
        "attempt": attempt.get("attempt"),
        "reward": attempt.get("reward"),
        "done": attempt.get("done"),
        "stopped_reason": attempt.get("stopped_reason", ""),
        "trajectory": str(attempt.get("trajectory_text", "")),
        "annotated_trajectory": compact_steps,
    }


def failure_score_lesson_prompt(cases: Sequence[Dict[str, Any]]) -> str:
    payload = []
    for item in cases:
        payload.append(
            {
                "session": item.get("session"),
                "first_failed_attempt": _compact_scored_attempt_for_failure_lesson(
                    item.get("first_failed_attempt") or {}
                ),
                "reflection": str(item.get("reflection", "")),
                "successful_attempt": _compact_scored_attempt_for_failure_lesson(
                    item.get("successful_attempt") or {}
                ),
            }
        )
    return f"""# Role
You are the InterCode SQL score learner.

# Input description
You are given 5 SQL tasks that failed first, then succeeded only after reflection.
Each first failed attempt and successful attempt includes compact scored step annotations.

# Rules
Focus on actions in the first failed attempt with low selected_distance that led to failure (reward < 1).
If a low-distance action looked close to the goal but caused or contributed to failure, extract one reusable score lesson explaining when future Action Scorer distance should be raised.
Use the reflection and successful attempt only to understand why the low-distance failed action was misleading.

# Output constraints
Do NOT output question-specific wording, literal table names, column names, or constant values in the lesson.
Return concise reusable scoring experience only.

# Output fields (per lesson)
1. before_state: abstract state before the misleading low-distance failed action
2. step: which low-distance failed action pattern was underestimated, stated abstractly
3. scoring_lesson: the scoring experience lesson that should guide future Action Scorer distance estimates.

# Output format
Output JSON array only. Each item must have exactly these fields:
[
  {{
    "before_state": "abstract before state",
    "step": "which low-distance failed action step/pattern",
    "scoring_lesson": "concise scoring lesson"
  }}
]

# Reflection-success cases
{json.dumps(payload, ensure_ascii=False)}

# Hard limit
Output at most 1 lesson per call (empty array if nothing is worth recording).
"""


def _reflection_success_case_id(case: Dict[str, Any]) -> str:
    basis = "\n".join(
        [
            str(case.get("session", "")),
            str((case.get("first_failed_attempt") or {}).get("trajectory_text", ""))[:1000],
            str(case.get("reflection", "")),
            str((case.get("successful_attempt") or {}).get("trajectory_text", ""))[:1000],
        ]
    )
    return "rs_" + hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]


def record_reflection_success_case(
    memory: Dict[str, Any],
    *,
    session: str,
    first_failed_attempt: Dict[str, Any],
    successful_attempt: Dict[str, Any],
    reflection: str,
    learn_every: int,
    success_reward: float = DEFAULT_SUCCESS_REWARD,
    max_cases: int = 50,
) -> None:
    if learn_every <= 0:
        return
    if not first_failed_attempt or not successful_attempt:
        return
    try:
        first_reward = float(first_failed_attempt.get("reward") or 0.0)
        final_reward = float(successful_attempt.get("reward") or 0.0)
    except Exception:
        return
    if first_reward >= success_reward or final_reward < success_reward:
        return
    case = {
        "session": session,
        "first_failed_attempt": {
            "attempt": first_failed_attempt.get("attempt"),
            "reward": first_reward,
            "trajectory_text": str(first_failed_attempt.get("trajectory_text", "")),
            "annotated_trajectory": list(first_failed_attempt.get("annotated_trajectory") or []),
        },
        "reflection": str(reflection or ""),
        "successful_attempt": {
            "attempt": successful_attempt.get("attempt"),
            "reward": successful_attempt.get("reward"),
            "trajectory_text": str(successful_attempt.get("trajectory_text", "")),
            "annotated_trajectory": list(successful_attempt.get("annotated_trajectory") or []),
        },
    }
    case["id"] = _reflection_success_case_id(case)
    stored = memory.setdefault("reflection_success_score_cases", [])
    seen = {str(item.get("id") or "") for item in stored if isinstance(item, dict)}
    if case["id"] not in seen:
        stored.append(case)
        del stored[:-max_cases]
        memory["reflection_success_score_case_total"] = int(memory.get("reflection_success_score_case_total", 0)) + 1
    total = int(memory.get("reflection_success_score_case_total", 0))
    if total < learn_every or total % learn_every != 0:
        return
    recent = stored[-learn_every:]
    raw, _ = llm_call(
        failure_score_lesson_prompt(recent),
        system="You derive concise reusable scoring lessons from low-distance actions that led to failed InterCode SQL attempts.",
        max_tokens=LEARNING_NOTES_MAX_TOKENS,
    )
    append_scoring_lessons(memory, parse_scoring_lessons(raw))


# ---------------------------------------------------------------------------
# 三层决策环
# ---------------------------------------------------------------------------

def fallback_candidate_actions(history: Dict[str, List[Any]], limit: int = 10) -> List[str]:
    if not history["actions"]:
        base = ["SHOW TABLES", "submit"]
    else:
        last_obs = history["observations"][-1] if history["observations"] else ""
        if "Error executing query" in str(last_obs) or "no such column" in str(last_obs) or "no such table" in str(last_obs):
            base = ["SHOW TABLES", "submit"]
        else:
            base = ["submit", "SHOW TABLES"]
    return _dedupe_preserve_order(base, limit)


def generate_candidate_actions(
    query: str,
    schema_hint: str,
    recent_trace: str,
    latest_observation: Any,
    memory_context: str,
    active_reflection: str,
    max_candidates: int,
    fallback_actions: Optional[Sequence[str]] = None,
    failed_trajectories: Sequence[str] = (),
) -> Tuple[List[str], str, Dict[str, Any]]:
    prompt = candidate_generation_prompt(
        query=query,
        schema_hint=schema_hint,
        recent_trace=recent_trace,
        latest_observation=latest_observation,
        memory_context=memory_context,
        active_reflection=active_reflection,
        max_candidates=max_candidates,
        failed_trajectories=failed_trajectories,
    )
    raw = ""
    usage: Dict[str, Any] = {}
    try:
        raw, usage = llm_call(
            prompt,
            system="You generate compact InterCode SQL action candidate sets as JSON.",
            max_tokens=CANDIDATE_GENERATION_MAX_TOKENS,
        )
        candidates = parse_candidate_actions(raw, limit=max_candidates)
        if candidates:
            return candidates, raw, usage
        fallback_prompt = (
            prompt
            + "\nThe previous output did not contain valid actions. Return a JSON array with one valid next action."
        )
        raw, usage = llm_call(
            fallback_prompt,
            system="You generate compact InterCode SQL action candidate sets as JSON.",
            max_tokens=CANDIDATE_GENERATION_MAX_TOKENS,
        )
        candidates = parse_candidate_actions(raw, limit=max_candidates)
        if candidates:
            return candidates, raw, usage
    except Exception as exc:
        raw = f"LLM_CANDIDATE_GENERATION_ERROR: {exc!r}\n{raw}"
        usage = {}
    fallback = _dedupe_preserve_order(fallback_actions or [], max_candidates)
    if fallback:
        return fallback, raw, usage
    return [], raw, usage


def score_candidate_actions(
    candidates: Sequence[str],
    query: str,
    schema_hint: str,
    recent_trace: str,
    latest_observation: Any,
    memory_context: str,
    active_reflection: str,
    failed_trajectories: Sequence[str] = (),
) -> Tuple[Dict[str, Dict[str, Any]], str, Dict[str, Any]]:
    try:
        raw, usage = llm_call(
            action_scoring_prompt(
                candidates=candidates,
                query=query,
                schema_hint=schema_hint,
                recent_trace=recent_trace,
                latest_observation=latest_observation,
                memory_context=memory_context,
                active_reflection=active_reflection,
                failed_trajectories=failed_trajectories,
            ),
            system="You estimate InterCode SQL goal distance for candidate actions and output JSON only.",
            max_tokens=ACTION_SCORING_MAX_TOKENS,
        )
    except Exception as exc:
        raw = f"LLM_ACTION_SCORING_ERROR: {exc!r}"
        usage = {}
    return normalize_action_scores_zscore_minmax(parse_action_scores(raw, candidates)), raw, usage


def parse_final_action_choice(raw: str, ranked_actions: Sequence[str]) -> Dict[str, Any]:
    ranked = list(ranked_actions or [])
    ranked_set = set(ranked)
    data = _loads_json_fragment(raw)
    action = ""
    decision_type = ""
    rationale = ""
    if isinstance(data, dict):
        action = _normalize_candidate_action(
            str(data.get("action") or data.get("selected_action") or data.get("next_action") or "")
        )
        decision_type = str(data.get("decision_type") or data.get("type") or "").strip().lower()
        rationale = str(data.get("rationale") or data.get("reason") or "").strip()
    elif isinstance(data, str):
        action = _normalize_candidate_action(data)
    if not action:
        action = _normalize_candidate_action(raw)
    fallback = False
    if action not in ranked_set:
        action = ranked[0] if ranked else action
        fallback = True
    if action.lower() == "submit":
        inferred_type = "submit"
    else:
        inferred_type = "execute"
    if decision_type not in {"execute", "submit"}:
        decision_type = inferred_type
    elif decision_type != inferred_type:
        decision_type = inferred_type
        fallback = True
    return {"action": action, "decision_type": decision_type, "rationale": rationale, "fallback": fallback}


def choose_final_action(
    ranked_actions: Sequence[str],
    scores: Dict[str, Dict[str, Any]],
    query: str,
    recent_trace: str,
    latest_observation: Any,
    active_reflection: str,
    failed_trajectories: Sequence[str] = (),
) -> Tuple[str, str, Dict[str, Any], bool, str, str]:
    try:
        raw, usage = llm_call(
            final_action_prompt(
                ranked_actions=ranked_actions,
                scores=scores,
                query=query,
                recent_trace=recent_trace,
                latest_observation=latest_observation,
                active_reflection=active_reflection,
                failed_trajectories=failed_trajectories,
            ),
            system="You select one listed InterCode SQL action decision and output JSON only.",
            max_tokens=FINAL_ACTION_MAX_TOKENS,
        )
        choice = parse_final_action_choice(raw, ranked_actions)
    except Exception as exc:
        raw = f"LLM_FINAL_ACTION_ERROR: {exc!r}"
        usage = {}
        choice = parse_final_action_choice("", ranked_actions)
        choice["fallback"] = True
    return (
        str(choice.get("action", "")),
        raw,
        usage,
        bool(choice.get("fallback", False)),
        str(choice.get("decision_type", "execute")),
        str(choice.get("rationale", "")),
    )


# ---------------------------------------------------------------------------
# 单次 Attempt
# ---------------------------------------------------------------------------

def run_policy_episode(
    *,
    env: SQLiteSqlEnv,
    task_index: int,
    out_dir: Path,
    attempt: int,
    max_turns: int,
    memory: Dict[str, Any],
    active_reflection: str = "",
    max_candidates: int = 10,
    failed_trajectories: Sequence[str] = (),
) -> Dict[str, Any]:
    observation, _ = env.reset(task_index)
    query = env.query
    schema_hint = schema_hint_from_record(env.record)
    history: Dict[str, List[Any]] = {
        "thoughts": [],
        "actions": [],
        "observations": [],
        "rewards": [],
        "valid_action": [],
    }
    model_events: List[Dict[str, Any]] = []
    final_reward = 0.0
    stopped_reason = ""
    latest_observation: Any = observation
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"task{task_index}_attempt{attempt}.jsonl"
    steps: List[Dict[str, Any]] = []
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for turn in range(1, max_turns + 1):
            if latest_observation is None:
                latest_observation = ""
            step_record = {
                "event": "step",
                "task_id": task_index,
                "attempt": attempt,
                "step": turn,
                "action": "(pending)",
                "observation": str(latest_observation)[:2000],
                "reward": 0.0,
                "done": False,
                "invalid": False,
                "schema_hint": schema_hint,
            }
            recent_trace = format_trajectory(query, history)
            mem_ctx = compact_memory_context(memory, recent_trace)
            model_candidates, raw_model_candidates, model_candidate_usage = generate_candidate_actions(
                query=query,
                schema_hint=schema_hint,
                recent_trace=recent_trace,
                latest_observation=latest_observation,
                memory_context=mem_ctx,
                active_reflection=active_reflection,
                max_candidates=max_candidates,
                fallback_actions=fallback_candidate_actions(history, max_candidates),
                failed_trajectories=failed_trajectories,
            )
            candidates = model_candidates
            raw_candidates = json.dumps(
                {"model_candidates": model_candidates, "model_raw": raw_model_candidates},
                ensure_ascii=False,
            )
            candidate_usage: Dict[str, Any] = {
                "source": "candidate_model_with_fallback",
                "model_usage": model_candidate_usage,
            }
            if not candidates:
                candidates = fallback_candidate_actions(history, max_candidates)
            scores, raw_scores, score_usage = score_candidate_actions(
                candidates=candidates,
                query=query,
                schema_hint=schema_hint,
                recent_trace=recent_trace,
                latest_observation=latest_observation,
                memory_context=mem_ctx,
                active_reflection=active_reflection,
                failed_trajectories=failed_trajectories,
            )
            ranked = rank_candidate_actions(candidates, scores)
            action, raw_action, action_usage, fallback, decision_type, decision_rationale = choose_final_action(
                ranked_actions=ranked,
                scores=scores,
                query=query,
                recent_trace=recent_trace,
                latest_observation=latest_observation,
                active_reflection=active_reflection,
                failed_trajectories=failed_trajectories,
            )
            if action == "submit":
                next_observation, reward, done, info = env.step("submit")
                valid_action = bool(info.get(ACTION_EXEC, True))
            else:
                next_observation, reward, done, info = env.step(action)
                valid_action = bool(info.get(ACTION_EXEC, True))
            next_observation = truncate_observation(next_observation)
            thought = f"Thought {turn}: BloomGoal selected the lowest-distance useful SQL action."
            history["thoughts"].append(thought)
            history["actions"].append(action)
            history["observations"].append(str(next_observation))
            history["rewards"].append(reward)
            history["valid_action"].append(valid_action)
            latest_observation = next_observation
            final_reward = float(reward)
            step_record["action"] = action
            step_record["reward"] = final_reward
            step_record["done"] = bool(done)
            step_record["invalid"] = not valid_action
            steps.append(step_record)
            jsonl.write(json.dumps(step_record, ensure_ascii=False) + "\n")
            model_record = {
                "event": "bloomgoal_model_decision",
                "task_id": task_index,
                "attempt": attempt,
                "turn": turn,
                "step": turn,
                "active_reflection": active_reflection,
                "candidate_actions": candidates,
                "raw_candidate_model": raw_candidates,
                "candidate_usage": candidate_usage,
                "scores": scores,
                "raw_score_model": raw_scores,
                "score_usage": score_usage,
                "ranked_actions": ranked,
                "raw_action_model": raw_action,
                "action_usage": action_usage,
                "normalized_action": action,
                "decision_type": decision_type,
                "decision_rationale": decision_rationale,
                "fallback_to_top_ranked": fallback,
            }
            model_events.append(model_record)
            jsonl.write(json.dumps(model_record, ensure_ascii=False) + "\n")
            jsonl.flush()
            if done:
                break

        if not done:
            next_observation, reward, done, info = env.step("submit")
            next_observation = truncate_observation(next_observation)
            valid_action = bool(info.get(ACTION_EXEC, True))
            history["thoughts"].append("EXCEEDED MAX TURNS: submit")
            history["actions"].append("submit")
            history["observations"].append(str(next_observation))
            history["rewards"].append(reward)
            history["valid_action"].append(valid_action)
            final_reward = float(reward)
            stopped_reason = "max_turns_exceeded"
            steps.append({
                "event": "step",
                "task_id": task_index,
                "attempt": attempt,
                "step": max_turns + 1,
                "action": "submit",
                "observation": str(next_observation)[:2000],
                "reward": final_reward,
                "done": bool(done),
                "invalid": not valid_action,
            })

    trajectory_text = format_trajectory(query, history)
    annotated_trajectory = build_annotated_success_trajectory(steps, model_events)
    return {
        "attempt": attempt,
        "reflection": active_reflection,
        "reward": final_reward,
        "done": bool(done),
        "num_steps": len(steps),
        "stopped_reason": stopped_reason,
        "steps_path": str(jsonl_path),
        "trajectory_text": trajectory_text,
        "annotated_trajectory": annotated_trajectory,
        "query": query,
        "turn_history": history,
    }


# ---------------------------------------------------------------------------
# 辅助：merge_reflection_candidates
# ---------------------------------------------------------------------------

def merge_reflection_candidates(
    memory: Dict[str, Any], generated: Sequence[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    verified_by_id: Dict[str, Dict[str, Any]] = {}
    for reflection in memory_reflections(memory, limit=50):
        verified_by_id[reflection["id"]] = reflection
    new_by_id: Dict[str, Dict[str, Any]] = {}
    for reflection in generated:
        normalized = normalize_reflection(reflection)
        if not normalized or normalized["id"] in verified_by_id or normalized["id"] in new_by_id:
            continue
        new_by_id[normalized["id"]] = normalized
    verified = list(verified_by_id.values())
    new_reflections = list(new_by_id.values())
    return verified, new_reflections, verified + new_reflections


# ---------------------------------------------------------------------------
# 任务级 run_bloomgoal_task (对齐 run_bloomgoal_session)
# ---------------------------------------------------------------------------

def is_solved(result: Dict[str, Any], success_reward: float) -> bool:
    return bool(result.get("done")) and float(result.get("reward") or 0.0) >= success_reward


def run_bloomgoal_task(
    *,
    data_path: Path,
    sqlite_root: Path,
    task_index: int,
    max_turns: int,
    retry_k: int,
    max_candidates: int,
    memory_path: Optional[Path] = DEFAULT_MEMORY_PATH,
    success_reward: float = DEFAULT_SUCCESS_REWARD,
    log_path: Optional[Path] = None,
    work_root: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    learn_every: int = 5,
    max_reflections: int = MAX_REFLECTIONS_DEFAULT,
) -> Dict[str, Any]:
    task_id = f"task_{task_index}"
    _reset_api_stats()
    task_start = time.perf_counter()
    out_dir, memory_path = resolve_run_paths(out_dir, log_path, memory_path)
    memory = load_memory(memory_path)
    env = SQLiteSqlEnv(data_path=data_path, sqlite_root=sqlite_root, work_root=work_root)
    attempts: List[Dict[str, Any]] = []
    failed_trajectories: List[str] = []
    reflection_events: List[Dict[str, Any]] = []
    generated_reflections: List[Dict[str, Any]] = []
    tried_ids: Set[str] = set()
    reflection_summary: List[Dict[str, Any]] = []
    retry_records: List[Dict[str, Any]] = []
    all_scored_reflections: Dict[str, Dict[str, Any]] = {}
    try:
        first = run_policy_episode(
            env=env,
            task_index=task_index,
            out_dir=out_dir,
            attempt=0,
            max_turns=max_turns,
            memory=memory,
            active_reflection="",
            max_candidates=max_candidates,
            failed_trajectories=failed_trajectories,
        )
        attempts.append(first)
        if is_solved(first, success_reward):
            append_success_trajectory(
                memory, task_id, first["reward"], first["trajectory_text"], first.get("annotated_trajectory", [])
            )
            maybe_update_learning_notes(memory, learn_every)
            save_memory(memory_path, memory)
        else:
            failed_trajectories.append(first["trajectory_text"])

        for retry_idx in range(1, retry_k + 1):
            if is_solved(attempts[-1], success_reward):
                break
            mem_ctx = compact_memory_context(memory, "\n\n".join(failed_trajectories))
            reflection_score_ctx = compact_reflection_score_context(memory)
            new_reflections, raw_generation, generation_usage = generate_failure_reflections(
                failed_trajectories, mem_ctx, retry_k,
                reflection_score_context=reflection_score_ctx,
            )
            generated_reflections.extend(new_reflections)
            verified_reflections, new_candidate_reflections, pool = merge_reflection_candidates(memory, generated_reflections)
            reflection_scores, raw_scoring, scoring_usage = score_reflections(
                reflections=pool,
                failed_trajectories=failed_trajectories,
                memory=memory,
            )
            scored_pool = rank_reflections_by_score(pool, reflection_scores)
            for ref in pool:
                rid = str(ref.get("id", ""))
                s = reflection_scores.get(rid, {})
                if rid in all_scored_reflections:
                    old = all_scored_reflections[rid]
                    n = int(old.get("_count", 1)) + 1
                    all_scored_reflections[rid] = {
                        "text": old.get("text", ""),
                        "score": (float(old.get("score", 0)) * (n - 1) + float(s.get("score", 0))) / n,
                        "relevance": (float(old.get("relevance", 0)) * (n - 1) + float(s.get("relevance", 0))) / n,
                        "actionability": (float(old.get("actionability", 0)) * (n - 1) + float(s.get("actionability", 0))) / n,
                        "conciseness": (float(old.get("conciseness", 0)) * (n - 1) + float(s.get("conciseness", 0))) / n,
                        "rationale": str(old.get("rationale", "")) + " | retry: " + str(s.get("rationale", "")),
                        "_count": n,
                    }
                else:
                    all_scored_reflections[rid] = {
                        "text": str(ref.get("text", "")),
                        "score": float(s.get("score", -1)),
                        "relevance": float(s.get("relevance", 0)),
                        "actionability": float(s.get("actionability", 0)),
                        "conciseness": float(s.get("conciseness", 0)),
                        "rationale": str(s.get("rationale", "")),
                        "_count": 1,
                    }
            selected, raw_selection, selection_usage = select_best_reflection(
                verified_reflections, new_candidate_reflections, failed_trajectories, tried_ids,
                reflection_scores=reflection_scores,
            )
            selected_score = reflection_scores.get(str(selected.get("id", "")), {}) if selected else {}
            reflection_event = {
                "event": "reflection_selection",
                "task_id": task_index,
                "retry_idx": retry_idx,
                "failed_trajectory_count": len(failed_trajectories),
                "generated_reflections": new_reflections,
                "raw_generation_model": raw_generation,
                "generation_usage": generation_usage,
                "reflection_scores": reflection_scores,
                "raw_scoring_model": raw_scoring,
                "scoring_usage": scoring_usage,
                "scored_pool": [
                    {"id": str(r.get("id")), "text": str(r.get("text", ""))[:800], "score": reflection_scores.get(str(r.get("id", "")), {}).get("score")}
                    for r in scored_pool
                ],
                "pool_size": len(pool),
                "verified_reflections": verified_reflections,
                "new_candidate_reflections": new_candidate_reflections,
                "tried_ids": sorted(tried_ids),
                "selected_reflection": selected,
                "selected_score": selected_score,
                "raw_selection_model": raw_selection,
                "selection_usage": selection_usage,
            }
            reflection_events.append(reflection_event)
            if not selected:
                break
            tried_ids.add(str(selected["id"]))
            retry_records.append({
                "failed_trajectory": failed_trajectories[-1],
                "reflection_text": str(selected["text"]),
                "score": float(selected_score.get("score", 0)),
                "relevance": float(selected_score.get("relevance", 0)),
                "actionability": float(selected_score.get("actionability", 0)),
                "conciseness": float(selected_score.get("conciseness", 0)),
                "rationale": str(selected_score.get("rationale", "")),
            })
            retry_result = run_policy_episode(
                env=env,
                task_index=task_index,
                out_dir=out_dir,
                attempt=retry_idx,
                max_turns=max_turns,
                memory=memory,
                active_reflection=str(selected["text"]),
                max_candidates=max_candidates,
                failed_trajectories=failed_trajectories,
            )
            attempts.append(retry_result)
            solved = is_solved(retry_result, success_reward)
            reflection_summary.append({
                "retry_idx": retry_idx,
                "reflection_text": str(selected["text"]),
                "score": float(selected_score.get("score", 0)),
                "relevance": float(selected_score.get("relevance", 0)),
                "actionability": float(selected_score.get("actionability", 0)),
                "conciseness": float(selected_score.get("conciseness", 0)),
                "solved": solved,
                "reward": float(retry_result.get("reward", 0)),
            })
            upsert_reflection(
                memory,
                str(selected["text"]),
                session=task_id,
                confirmed=solved,
                success=solved,
                max_reflections=max_reflections,
            )
            if solved:
                append_success_trajectory(
                    memory, task_id, retry_result["reward"], retry_result["trajectory_text"],
                    retry_result.get("annotated_trajectory", []),
                )
                record_reflection_success_case(
                    memory,
                    session=task_id,
                    first_failed_attempt=first,
                    successful_attempt=retry_result,
                    reflection=str(selected["text"]),
                    learn_every=learn_every,
                    success_reward=success_reward,
                )
                maybe_update_learning_notes(memory, learn_every)
                save_memory(memory_path, memory)
                break
            failed_trajectories.append(retry_result["trajectory_text"])
            save_memory(memory_path, memory)

        best_attempt = max(attempts, key=lambda item: float(item.get("reward") or 0.0))
        solved = any(is_solved(item, success_reward) for item in attempts)
        if solved and retry_records:
            success_attempt = next((a for a in attempts if is_solved(a, success_reward)), None)
            if success_attempt:
                learn_reflection_scoring_from_successful_task(
                    memory=memory,
                    retry_records=retry_records,
                    successful_trajectory=success_attempt.get("trajectory_text", ""),
                    all_scored_reflections=all_scored_reflections,
                )
                save_memory(memory_path, memory)
        result = {
            "environment": env.name,
            "dataset": str(data_path),
            "algorithm": "BloomGoal_SQLite",
            "model": MODEL,
            "base_url": BASE_URL,
            "use_proxy": USE_PROXY,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "task_id": task_index,
            "task_index": task_index,
            "query": best_attempt.get("query") or first.get("query") or getattr(env, "query", ""),
            "db": getattr(env, "current_db", ""),
            "reward": float(best_attempt.get("reward") or 0.0),
            "done": any(bool(item.get("done")) for item in attempts),
            "solved": solved,
            "max_turns": max_turns,
            "retry_k": retry_k,
            "max_candidates": max_candidates,
            "memory_path": str(memory_path),
            "learn_every": learn_every,
            "max_reflections": max_reflections,
            "num_attempts": len(attempts),
            "num_steps": sum(int(item.get("num_steps", 0)) for item in attempts),
            "best_attempt": int(best_attempt.get("attempt", 0)),
            "elapsed_seconds": round(time.perf_counter() - task_start, 3),
            "api_stats": _snapshot_api_stats(),
            "turn_history": best_attempt.get("turn_history", {}),
            "summary": {
                "max_reward": float(best_attempt.get("reward") or 0.0),
                "turns_taken": int(best_attempt.get("num_steps", 0)),
                "turns_max": max_turns,
                "num_attempts": len(attempts),
                "best_attempt": attempts.index(best_attempt),
            },
            "attempts": attempts,
            "reflection_summary": reflection_summary,
            "reflection_events": reflection_events,
        }
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        save_memory(memory_path, memory)
        return result
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Eval / main
# ---------------------------------------------------------------------------

def run_bloomgoal_eval(
    *,
    data_path: Path,
    sqlite_root: Path,
    start_index: int,
    limit: int,
    max_turns: int,
    retry_k: int,
    max_candidates: int,
    log_path: Optional[Path] = None,
    memory_path: Optional[Path] = DEFAULT_MEMORY_PATH,
    work_root: Optional[Path] = None,
    learn_every: int = 5,
    max_reflections: int = MAX_REFLECTIONS_DEFAULT,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    records = json.loads(Path(data_path).read_text(encoding="utf-8"))
    stop = min(len(records), start_index + limit)
    results: Dict[str, Any] = {}
    out_root, memory_path = resolve_run_paths(out_dir, log_path, memory_path)
    for index in range(start_index, stop):
        result = run_bloomgoal_task(
            data_path=data_path,
            sqlite_root=sqlite_root,
            task_index=index,
            max_turns=max_turns,
            retry_k=retry_k,
            max_candidates=max_candidates,
            memory_path=memory_path,
            log_path=None,
            work_root=work_root,
            out_dir=out_root / f"task_{index}",
            learn_every=learn_every,
            max_reflections=max_reflections,
        )
        results[str(index)] = result
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BloomGoal-Retrieval over InterCodeSQL SQLite tasks.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--sqlite-root", required=True)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--retry-k", type=int, default=int(os.environ.get("BLOOMGOAL_RETRY_K", "2")))
    parser.add_argument("--max-candidates", type=int, default=int(os.environ.get("BLOOMGOAL_MAX_CANDIDATES", "8")))
    parser.add_argument(
        "--memory-path",
        help="Explicit memory override; default: result directory/BloomGoal_SQL_memory.json",
    )
    parser.add_argument("--log-path")
    parser.add_argument("--work-root")
    parser.add_argument("--api-env")
    parser.add_argument("--out-dir")
    parser.add_argument("--success-reward", type=float, default=DEFAULT_SUCCESS_REWARD)
    parser.add_argument("--learn-every", type=int, default=int(os.environ.get("BLOOMGOAL_LEARN_EVERY", "5")))
    parser.add_argument("--max-reflections", type=int, default=int(os.environ.get("BLOOMGOAL_MAX_REFLECTIONS", str(MAX_REFLECTIONS_DEFAULT))))
    args = parser.parse_args()
    if args.api_env:
        load_env_file(Path(args.api_env))
    load_env_file(Path(".api_env"))

    if args.limit == 1:
        out_dir = Path(args.out_dir) if args.out_dir else None
        result = run_bloomgoal_task(
            data_path=Path(args.data_path),
            sqlite_root=Path(args.sqlite_root),
            task_index=args.task_index,
            max_turns=args.max_turns,
            retry_k=args.retry_k,
            max_candidates=args.max_candidates,
            memory_path=Path(args.memory_path) if args.memory_path else None,
            log_path=Path(args.log_path) if args.log_path else None,
            work_root=Path(args.work_root) if args.work_root else None,
            out_dir=out_dir,
            learn_every=args.learn_every,
            max_reflections=args.max_reflections,
        )
        print(
            json.dumps(
                {
                    "reward": result["reward"],
                    "done": result["done"],
                    "solved": result["solved"],
                    "task_id": result["task_id"],
                    "attempts": len(result["attempts"]),
                    "best_attempt": result["best_attempt"],
                },
                ensure_ascii=False,
            )
        )
        return

    results = run_bloomgoal_eval(
        data_path=Path(args.data_path),
        sqlite_root=Path(args.sqlite_root),
        start_index=args.task_index,
        limit=args.limit,
        max_turns=args.max_turns,
        retry_k=args.retry_k,
        max_candidates=args.max_candidates,
        log_path=Path(args.log_path) if args.log_path else None,
        memory_path=Path(args.memory_path) if args.memory_path else None,
        work_root=Path(args.work_root) if args.work_root else None,
        learn_every=args.learn_every,
        max_reflections=args.max_reflections,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )
    solved = sum(1 for item in results.values() if float(item.get("reward") or 0.0) >= 1.0)
    print(json.dumps({"episodes": len(results), "solved": solved, "log_path": args.log_path}, ensure_ascii=False))


if __name__ == "__main__":
    main()
