import ast
import importlib
import inspect
import json
import re
import sqlite3
import sys
import tempfile
import unittest
import warnings
from contextlib import closing
from pathlib import Path
from unittest import mock
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webshop"))
sys.path.insert(0, str(ROOT / "intercode_sql"))


class ReleaseRunnerTests(unittest.TestCase):
    def test_alfworld_distance_matches_paper_formula(self):
        names = {
            "extract_goal_entities", "extract_action_entities", "lexical_reduction",
            "parse_action_distance_json", "llm_action_reduction_scores", "compute_action_distances",
        }
        for filename in ("bloom_dist.py", "bloom_dist_zero_shot.py", "train.py"):
            with self.subTest(filename=filename):
                source = ROOT / "alfworld" / filename
                tree = ast.parse(source.read_text(encoding="utf-8"))
                selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
                prompts = []

                def llm2(prompt, **_kwargs):
                    prompts.append(prompt)
                    return json.dumps({"take apple from table": 1.0})

                scope = {
                    "re": re, "json": json, "List": List, "Dict": Dict,
                    "OpenAI": object, "StepRecord": object, "llm2": llm2,
                    "format_trajectory_like_prompt": lambda _trajectory: "",
                }
                exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), scope)
                goal = "Task: put apple in bowl"
                action = "take apple from table"
                self.assertEqual(scope["lexical_reduction"](goal, action), 0.5)
                self.assertEqual(scope["parse_action_distance_json"](
                    json.dumps({action: 1.5}), [action]
                )[action], 1.0)
                distances = scope["compute_action_distances"](
                    [action], "", goal, [], None, "test-model", 100
                )
                self.assertEqual(distances[action], 0.25)
                self.assertIn("reduction score in [0, 1]", prompts[0])

    def test_alfworld_has_its_own_directory(self):
        for name in (
            "bloom_dist.py", "bloom_dist_zero_shot.py", "train.py", "train_max.py",
            "base_config.yaml", "requirement.txt", "prompts/alfworld_3prompts.json",
        ):
            with self.subTest(name=name):
                self.assertTrue((ROOT / "alfworld" / name).is_file())
                self.assertFalse((ROOT / name).exists())
        self.assertFalse((ROOT / "API_KEY.txt").exists())

    def test_webshop_loads_api_env_before_server_configuration(self):
        source = (ROOT / "webshop" / "bloomgoal_webshop_runner.py").read_text(encoding="utf-8")
        self.assertGreater(
            source.index("import webshop_environment as base"),
            source.index("if API_ENV.exists():"),
        )

    def test_webshop_runner_imports_without_private_base(self):
        module = importlib.import_module("bloomgoal_webshop_runner")
        self.assertTrue(callable(module.run_bloomgoal_session))

    def test_webshop_defaults_match_paper(self):
        module = importlib.import_module("bloomgoal_webshop_runner")
        with mock.patch.object(module, "run_bloomgoal_session") as run:
            with mock.patch.object(sys, "argv", ["runner", "--session", "test_0"]):
                module.main()
        options = run.call_args.kwargs
        self.assertEqual(options["max_steps"], 50)
        self.assertEqual(options["retry_k"], 2)
        self.assertEqual(options["max_candidates"], 3)

    def test_intercode_runners_import_without_react_baseline(self):
        base = importlib.import_module("sqlite_sql.bloomgoal_sqlite_runner")
        retrieval = importlib.import_module("sqlite_sql.bloomgoal_retrieval_sqlite_runner")
        self.assertTrue(callable(base.run_bloomgoal_task))
        self.assertTrue(callable(retrieval.run_bloomgoal_task))

    def test_intercode_defaults_match_paper(self):
        for name in ("bloomgoal_sqlite_runner", "bloomgoal_retrieval_sqlite_runner"):
            with self.subTest(name=name):
                module = importlib.import_module(f"sqlite_sql.{name}")
                result = {
                    "reward": 0.0,
                    "done": False,
                    "solved": False,
                    "task_id": "task_0",
                    "attempts": [],
                    "best_attempt": 0,
                }
                with mock.patch.object(module, "run_bloomgoal_task", return_value=result) as run:
                    with mock.patch.object(
                        sys,
                        "argv",
                        ["runner", "--data-path", "tasks.json", "--sqlite-root", "databases"],
                    ):
                        module.main()
                options = run.call_args.kwargs
                self.assertEqual(options["max_turns"], 15)
                self.assertEqual(options["retry_k"], 2)
                self.assertEqual(options["max_candidates"], 8)

    def test_base_window_and_retrieval_memory_are_distinct(self):
        base = importlib.import_module("sqlite_sql.bloomgoal_sqlite_runner")
        retrieval = importlib.import_module("sqlite_sql.bloomgoal_retrieval_sqlite_runner")
        base_memory = {"scoring_lessons": []}
        retrieval_memory = {"scoring_lessons": []}
        for index in range(12):
            lesson = {
                "before_state": f"state {index} target{index}",
                "step": f"execute SELECT {index}",
                "scoring_lesson": f"prefer target{index}",
            }
            base.append_scoring_lessons(base_memory, [lesson])
            retrieval.append_scoring_lessons(retrieval_memory, [lesson])
        self.assertEqual(len(base_memory["scoring_lessons"]), 10)
        self.assertEqual(len(retrieval_memory["scoring_lessons"]), 12)
        selected = retrieval.select_relevant_scoring_lessons(
            retrieval_memory, "state 11 target11", limit=1
        )
        self.assertEqual(selected[0]["before_state"], "state 11 target11")
        base_reflections = {"reflection_scoring_lessons": []}
        retrieval_reflections = {"reflection_scoring_lessons": []}
        for index in range(12):
            lesson = {"direction": "overrated", "scoring_lesson": f"reflection rule {index}"}
            base.append_reflection_scoring_lessons(base_reflections, [lesson])
            retrieval.append_reflection_scoring_lessons(retrieval_reflections, [lesson])
        self.assertEqual(len(base_reflections["reflection_scoring_lessons"]), 10)
        self.assertEqual(len(retrieval_reflections["reflection_scoring_lessons"]), 12)

    def test_base_candidate_default_is_eight(self):
        base = importlib.import_module("sqlite_sql.bloomgoal_sqlite_runner")
        self.assertEqual(inspect.signature(base.run_policy_episode).parameters["max_candidates"].default, 8)

    def test_webshop_scoring_lessons_use_ten_entry_windows(self):
        module = importlib.import_module("bloomgoal_webshop_runner")
        action_memory = {"scoring_lessons": []}
        reflection_memory = {"reflection_scoring_lessons": []}
        for index in range(12):
            module.append_scoring_lessons(
                action_memory,
                [{
                    "before_state": f"product state {index}",
                    "step": f"click item {index}",
                    "scoring_lesson": f"action rule {index}",
                }],
            )
            module.append_reflection_scoring_lessons(
                reflection_memory,
                [{"direction": "underestimated", "scoring_lesson": f"reflection rule {index}"}],
            )
        self.assertEqual(len(action_memory["scoring_lessons"]), 10)
        self.assertEqual(len(reflection_memory["reflection_scoring_lessons"]), 10)

    def test_webshop_environment_keeps_navigation_and_options(self):
        adapter = importlib.import_module("webshop_environment")
        env = adapter.webshopEnv()

        def page(**state):
            page_type = state["page_type"]
            if page_type == "search":
                return "result", {"asins": ["ITEM1"]}
            if page_type == "item":
                return "item", {"option_types": {"blue": "Color"}}
            if page_type == "end":
                return "done", {"reward": 1.0}
            return "search", {}

        with mock.patch.object(adapter, "webshop_text", side_effect=page):
            self.assertEqual(env.step("task", "reset")[0], "search")
            self.assertEqual(env.step("task", "search[blue mug]")[0], "result")
            self.assertEqual(env.step("task", "click[ITEM1]")[0], "item")
            self.assertEqual(env.step("task", "click[blue]")[0], "You have clicked blue.")
            observation, reward, done = env.step("task", "click[Buy Now]")
        self.assertEqual((observation, reward, done), ("done", 1.0, True))
        self.assertEqual(env.sessions["task"]["options"], {"Color": "blue"})

    def test_sqlite_environment_executes_and_scores_a_task(self):
        env_module = importlib.import_module("sqlite_sql.env")
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temp:
            root = Path(temp)
            db_dir = root / "toy"
            db_dir.mkdir()
            with closing(sqlite3.connect(db_dir / "toy.sqlite")) as connection:
                connection.execute("CREATE TABLE facts (value INTEGER)")
                connection.execute("INSERT INTO facts VALUES (42)")
                connection.commit()
            tasks = root / "tasks.json"
            tasks.write_text(
                json.dumps([{"db": "toy", "query": "Find the value", "gold": "SELECT value FROM facts"}]),
                encoding="utf-8",
            )
            with env_module.SQLiteSqlEnv(data_path=tasks, sqlite_root=root, work_root=root) as env:
                env.reset(0)
                observation, reward, done, _info = env.step("SELECT value FROM facts")
                self.assertEqual(observation, [(42,)])
                self.assertEqual((reward, done), (0, False))
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    _observation, reward, done, _info = env.step("submit")
                self.assertEqual((reward, done), (1.0, True))

    def test_sql_action_parser_handles_nested_brackets(self):
        support = importlib.import_module("sqlite_sql.runner_support")
        action, found = support.parse_react_action("Action 1: execute[SELECT '[x]' AS value;]")
        self.assertTrue(found)
        self.assertEqual(action, "SELECT '[x]' AS value")


if __name__ == "__main__":
    unittest.main()
