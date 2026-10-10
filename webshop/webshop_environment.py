"""Minimal WebShop environment adapter used by the BloomGoal runner.

The runner installs its URL-safe page parser as ``webshop_text`` on import.
The WebShop benchmark server itself is a separate dependency.
"""

import os

from bs4 import BeautifulSoup
from bs4.element import Comment


WEBSHOP_URL = os.environ.get("WEBSHOP_URL", "http://127.0.0.1:3000").rstrip("/")
ACTION_TO_TEMPLATE = {
    "Description": "description_page.html",
    "Features": "features_page.html",
    "Reviews": "review_page.html",
    "Attributes": "attributes_page.html",
}


def clean_str(value):
    return value.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


def tag_visible(element):
    ignored = {"style", "script", "head", "title", "meta", "[document]"}
    return element.parent.name not in ignored and not isinstance(element, Comment)


def webshop_text(**_kwargs):
    raise RuntimeError("BloomGoal must install its WebShop page parser before use")


class webshopEnv:
    def __init__(self):
        self.sessions = {}

    def step(self, session, action):
        done = False
        observation_override = None
        if action == "reset":
            self.sessions[session] = {"session": session, "page_type": "init"}
        elif action.startswith("think["):
            pass
        elif action.startswith("search["):
            assert self.sessions[session]["page_type"] == "init"
            query = action[7:-1]
            self.sessions[session] = {
                "session": session,
                "page_type": "search",
                "query_string": query,
                "page_num": 1,
            }
        elif action.startswith("click["):
            button = action[6:-1]
            state = self.sessions[session]
            page_type = state["page_type"]
            if button == "Buy Now":
                assert page_type == "item"
                state["page_type"] = "end"
                done = True
            elif button == "Back to Search":
                assert page_type in ("search", "item_sub", "item")
                self.sessions[session] = {"session": session, "page_type": "init"}
            elif button == "Next >":
                raise AssertionError("Next > is not supported by this adapter")
            elif button == "< Prev":
                assert page_type in ("search", "item_sub", "item")
                if page_type == "search":
                    raise AssertionError("No previous search page")
                if page_type == "item_sub":
                    state["page_type"] = "item"
                else:
                    state["page_type"] = "search"
                    state["options"] = {}
            elif button in ACTION_TO_TEMPLATE:
                assert page_type == "item"
                state["page_type"] = "item_sub"
                state["subpage"] = button
            elif page_type == "search":
                assert button in state.get("asins", []), (button, state.get("asins", []))
                state["page_type"] = "item"
                state["asin"] = button
            elif page_type == "item":
                assert button in state.get("option_types", {}), (button, state.get("option_types", {}))
                option_type = state["option_types"][button]
                state.setdefault("options", {})[option_type] = button
                observation_override = f"You have clicked {button}."
            else:
                raise AssertionError(f"Invalid click on {page_type}: {button}")
        else:
            raise AssertionError(action)

        observation, info = webshop_text(**self.sessions[session])
        if observation_override is not None:
            observation = observation_override
        self.sessions[session].update(info)
        return observation, info.get("reward", 0.0), done


def allowed_hint_for(env, session):
    state = env.sessions.get(session, {})
    page_type = state.get("page_type")
    if page_type == "init":
        return (
            "Current page is the initial WebShop search page. "
            "The only valid next action is search[concise product query]. "
            "Do not output think[...] or click[...] on this page."
        )
    if page_type == "search":
        asins = state.get("asins", [])
        return (
            "Current page is a search results page. Valid product clicks are: "
            + ", ".join(f"click[{asin}]" for asin in asins)
            + ". You may output think[...] before choosing, but do not click labels not shown here."
        )
    if page_type == "item":
        options = list(state.get("option_types", {}).keys())
        labels = options + ["Description", "Features", "Reviews", "Buy Now", "< Prev"]
        return (
            "Current page is an item page. Valid click labels are: "
            + ", ".join(f"click[{label}]" for label in labels[:60])
            + ". Select required options before Buy Now when needed."
        )
    if page_type == "item_sub":
        return "Current page is an item subpage. Valid next actions include click[< Prev] or think[...]."
    return "Output exactly one valid ReAct WebShop action."
