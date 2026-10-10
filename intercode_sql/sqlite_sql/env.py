import json
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from sqlite_sql.mysql_compat import format_description_rows, translate_mysql_probe
from sqlite_sql.reward import sql_iou_reward


ACTION_EXEC = "action_executed"
AGENT_OBS = "agent_obs"
EVAL_OBS = "eval_obs"
CORRUPT_GOLD = "corrupt_gold"
REWARD = "reward"


class SQLiteSqlEnv:
    name = "ic_sql_sqlite"

    def __init__(self, data_path: Path, sqlite_root: Path, work_root: Optional[Path] = None, verbose: bool = False):
        self.data_path = Path(data_path)
        self.sqlite_root = Path(sqlite_root)
        self.work_root = Path(work_root) if work_root else None
        self.verbose = verbose
        self.records = json.loads(self.data_path.read_text(encoding="utf-8"))
        self.tmpdir: Optional[tempfile.TemporaryDirectory] = None
        self.cnx: Optional[sqlite3.Connection] = None
        self.cur: Optional[sqlite3.Cursor] = None
        self.info: Dict[str, Any] = {}
        self.observation: Any = None
        self.reward = 0.0
        self.record: Dict[str, Any] = {}
        self.query = ""
        self.gold = ""
        self.current_db = ""
        self.current_db_path: Optional[Path] = None

    def __len__(self) -> int:
        return len(self.records)

    def _source_db_path(self, db_name: str) -> Path:
        candidates = [
            self.sqlite_root / db_name / f"{db_name}.sqlite",
            self.sqlite_root / db_name / f"{db_name}.db",
            self.sqlite_root / f"{db_name}.sqlite",
            self.sqlite_root / f"{db_name}.db",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"SQLite database for `{db_name}` not found under {self.sqlite_root}")

    def _close_connection(self) -> None:
        if self.cur is not None:
            self.cur.close()
            self.cur = None
        if self.cnx is not None:
            self.cnx.close()
            self.cnx = None

    def _cleanup_workdir(self) -> None:
        if self.tmpdir is not None:
            self.tmpdir.cleanup()
            self.tmpdir = None

    def reset(self, index: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        self._close_connection()
        self._cleanup_workdir()
        idx = 0 if index is None else index
        self.record = self.records[idx]
        self.query = self.record["query"]
        self.gold = self.record.get("gold", "")
        self.current_db = self.record["db"]

        src = self._source_db_path(self.current_db)
        if self.work_root:
            self.work_root.mkdir(parents=True, exist_ok=True)
            self.tmpdir = tempfile.TemporaryDirectory(prefix=f"{self.current_db}_", dir=str(self.work_root))
        else:
            self.tmpdir = tempfile.TemporaryDirectory(prefix=f"{self.current_db}_")

        temp_path = Path(self.tmpdir.name)
        self.current_db_path = temp_path / src.name
        shutil.copy2(src, self.current_db_path)
        self.cnx = sqlite3.connect(self.current_db_path)
        self.cur = self.cnx.cursor()
        self.info = {}
        self.observation = self.query
        self.reward = 0.0
        return self.observation, self.info

    def _ensure_ready(self) -> sqlite3.Cursor:
        if self.cur is None or self.cnx is None:
            raise RuntimeError("Environment must be reset before executing actions")
        return self.cur

    def step(self, action: str) -> Tuple[Any, float, bool, Dict[str, Any]]:
        if action == "skip":
            return "skipped", 0, True, {}
        if action.strip().lower() == "submit":
            reward, info = self.get_reward()
            self.info = info
            return self.observation, reward, True, info
        self.exec_action(action)
        return self.observation, 0, False, self.info

    def exec_action(self, action: str) -> None:
        cur = self._ensure_ready()
        self.info = {}
        try:
            translated = translate_mysql_probe(action)
            if translated.db_name and translated.db_name != self.current_db:
                raise sqlite3.OperationalError(f"Unknown database '{translated.db_name}'")
            if translated.kind == "use":
                self.observation = []
                self.info[ACTION_EXEC] = True
                return

            cur.execute(translated.sql)
            if translated.kind == "describe":
                rows = cur.fetchall()
                if not rows:
                    raise sqlite3.OperationalError(f"Table '{self.current_db}.{translated.table_name}' doesn't exist")
                self.observation = format_description_rows(rows)
            elif cur.description is not None:
                self.observation = cur.fetchall()
            else:
                self.cnx.commit()
                self.observation = []
            self.info[ACTION_EXEC] = True
        except Exception as err:
            self.observation = f"Error executing query: {err}"
            self.info[ACTION_EXEC] = False

    def get_reward(self) -> Tuple[float, Dict[str, Any]]:
        info: Dict[str, Any] = {AGENT_OBS: self.observation}
        try:
            cur = self._ensure_ready()
            cur.execute(translate_mysql_probe(self.gold).sql)
            eval_obs = cur.fetchall() if cur.description is not None else []
            info[CORRUPT_GOLD] = False
        except Exception as err:
            eval_obs = f"Error executing query: {err}"
            info[CORRUPT_GOLD] = True
        reward, reward_info = sql_iou_reward(self.observation, eval_obs)
        info.update(reward_info)
        info[EVAL_OBS] = str(info[EVAL_OBS])
        info[AGENT_OBS] = str(info[AGENT_OBS])
        self.reward = reward
        return reward, info

    def close(self) -> None:
        self._close_connection()
        self._cleanup_workdir()

    def __enter__(self) -> "SQLiteSqlEnv":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
