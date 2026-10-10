"""Prompt example and parsing helpers shared by the BloomGoal SQL runners."""

import os
import re
from pathlib import Path
from typing import Any, Optional, Tuple


REACT_SQL_DEMO = """Question: What are the names and grades for each high schooler?
Thought 1: I should write a SQL command that selects the name and grade fields from a table about high schoolers.
Action 1: execute[SELECT name, grade FROM high_schoolers]
Observation 1: Error executing query: no such table: high_schoolers
Thought 2: The table does not exist, so I need to check what tables are available.
Action 2: execute[SHOW TABLES]
Observation 2: [('friend',), ('highschooler',), ('likes',)]
Thought 3: I should inspect the highschooler table.
Action 3: execute[DESC highschooler]
Observation 3: [('ID', 'int', 'NO', 'PRI', None, 'auto_increment'), ('name', 'text', 'YES', '', None, ''), ('grade', 'int', 'YES', '', None, '')]
Thought 4: The table has name and grade columns, so I can answer.
Action 4: execute[SELECT name, grade FROM highschooler]
Observation 4: [('John', 12), ('Haley', 10)]
Thought 5: The output answers the question.
Action 5: submit
"""


def strip_terminal_semicolon(sql: str) -> str:
    sql = (sql or "").strip()
    return sql[:-1].strip() if sql.endswith(";") else sql


def extract_last_execute_payload(text: str) -> Optional[str]:
    marker = "execute["
    start = text.lower().rfind(marker)
    if start < 0:
        return None
    payload_start = start + len(marker)
    depth = 1
    in_single = False
    in_double = False
    escaped = False
    for index in range(payload_start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            continue
        if in_single or in_double:
            continue
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[payload_start:index]
    return text[payload_start:]


def parse_react_action(text: str) -> Tuple[str, bool]:
    text = (text or "").strip()
    if text.lower() == "submit" or re.search(r"(?im)^Action\s+\d+:\s*submit\s*$", text):
        return "submit", True
    payload = extract_last_execute_payload(text)
    if payload is not None:
        return strip_terminal_semicolon(payload), True
    return text, False


def truncate_observation(observation: Any, max_chars: int = 600, max_rows: int = 25) -> Any:
    if isinstance(observation, str) and len(observation) > max_chars:
        return observation[:max_chars]
    if isinstance(observation, list) and len(observation) > max_rows:
        return observation[:max_rows]
    return observation


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())
