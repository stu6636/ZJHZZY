import math
from collections import Counter
from itertools import chain, groupby
from operator import itemgetter
from typing import Any, Dict, List, Tuple

try:
    from scipy.stats import kendalltau
except Exception:  # pragma: no cover - fallback for minimal installs.
    kendalltau = None


def get_intersect_items(items: List[str], counts: Dict[str, int]) -> List[str]:
    result = []
    for item in items:
        if item in counts:
            counts[item] -= 1
            if counts[item] == 0:
                del counts[item]
            result.append(item)
    return result


def sql_iou_reward(agent_obs: Any, eval_obs: Any) -> Tuple[float, Dict[str, Any]]:
    info: Dict[str, Any] = {"agent_obs": agent_obs, "eval_obs": eval_obs}
    if not isinstance(agent_obs, list) or not isinstance(eval_obs, list):
        info["reward"] = 0.0
        return 0.0, info

    list_agent = [str(x) for x in agent_obs]
    list_eval = [str(x) for x in eval_obs]
    dist_agent = Counter(list_agent)
    dist_eval = Counter(list_eval)
    intersection = dist_agent & dist_eval

    get_key, get_val = itemgetter(0), itemgetter(1)
    merged_data = sorted(chain(dist_agent.items(), dist_eval.items()), key=get_key)
    union = {k: max(map(get_val, g)) for k, g in groupby(merged_data, key=get_key)}

    if len(union) == 0:
        reward = 1
    else:
        total_intersect = sum(v for _, v in intersection.items())
        total_union = sum(v for _, v in union.items())
        reward = total_intersect * 1.0 / total_union
        if len(intersection) > 1 and kendalltau is not None:
            list_agent_intx = get_intersect_items(list_agent.copy(), intersection.copy())
            list_eval_intx = get_intersect_items(list_eval.copy(), intersection.copy())
            rank = {value: index for index, value in enumerate(sorted(set(list_agent_intx + list_eval_intx)))}
            sort_correlation = kendalltau(
                [rank[value] for value in list_agent_intx],
                [rank[value] for value in list_eval_intx],
                nan_policy="omit",
            ).statistic
            if sort_correlation is not None and math.isfinite(sort_correlation):
                reward = round(sort_correlation * reward, 2)
    info["reward"] = reward
    return reward, info
