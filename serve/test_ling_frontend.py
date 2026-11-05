"""serve/test_ling_frontend.py - Ling-3 in the frontend: its chat template (tools/ling/chat_template.jinja, the one its
GGUF carries) rendered with the server's thinking switch, tools and tool results, the model's reasoning, answers and
GLM-style tool calls parsed from any split of the stream, and each turn's prompt a prefix of the next one's (the
engine reads on from its state only then).

    python -m unittest serve.test_ling_frontend
"""
from __future__ import annotations

import json
import random
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate, OutputParser, effort_kwargs, openai_to_messages

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "tools" / "ling" / "chat_template.jinja"
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "The weather of a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                   "required": ["city"]}}}]
GEN = "<role>ASSISTANT</role>"


def events(out: str, thinking=True, seed=0):
    """The parser's events for `out` fed in random pieces: (reasoning, content, tool calls)."""
    rng = random.Random(seed)
    p = OutputParser(thinking=thinking, tools=[t["function"] for t in TOOLS], stream_tools=True)
    evs, i = [], 0
    while i < len(out):
        n = rng.randint(1, 9)
        evs += p.feed(out[i:i + n])
        i += n
    evs += p.finish()
    return ("".join(e.text for e in evs if e.kind == "reasoning"), "".join(e.text for e in evs if e.kind == "content"),
            [e.call for e in evs if e.kind == "tool_call"])


@unittest.skipUnless(TEMPLATE.exists(), "tools/ling/chat_template.jinja (make_pack.py's, from the GGUF) not here")
class Template(unittest.TestCase):
    def setUp(self):
        self.t = ChatTemplate(TEMPLATE)

    def test_family(self):
        self.assertTrue(self.t.ling)
        self.assertFalse(self.t.minimax or self.t.deepseek or self.t.glm_effort or getattr(self.t, "inkling", False))

    def test_thinking(self):
        msgs = [{"role": "user", "content": "Hi"}]
        on = self.t.render(msgs)
        self.assertTrue(on.startswith("<role>SYSTEM</role>detailed thinking on<|role_end|>"), on[:80])
        self.assertTrue(on.endswith("<role>HUMAN</role>Hi<|role_end|>" + GEN + "\n<think>"), on[-80:])
        off = self.t.render(msgs, **effort_kwargs("none"))
        self.assertIn("detailed thinking off<|role_end|>", off)
        self.assertTrue(off.endswith(GEN + "\n<think></think>"), off[-60:])
        self.assertEqual(self.t.render(msgs, enable_thinking=False), off)
        # (the effort words mean nothing to it: thinking stays on)
        self.assertEqual(self.t.render(msgs, **effort_kwargs("low")), on)
        sysm = [{"role": "system", "content": "Be brief."}] + msgs
        self.assertTrue(self.t.render(sysm).startswith("<role>SYSTEM</role>Be brief.\ndetailed thinking on<|role_end|>"))

    def test_tools(self):
        req = {"messages": [{"role": "user", "content": "Weather in Paris?"}], "tools": TOOLS}
        messages, tools, kwargs = openai_to_messages(req)
        out = self.t.render(messages, tools, **kwargs)
        self.assertIn("<tools>\n" + json.dumps(TOOLS[0], ensure_ascii=False) + "\n</tools>", out)
        self.assertIn("<tool_call>{function-name}\n<arg_key>{arg-key-1}</arg_key>", out)

    def test_history(self):
        req = {"messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "reasoning_content": "Call the tool.", "content": "",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\", \"days\": 2}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "{\"temp\": 21}"},
            {"role": "assistant", "content": "21 degrees."},
            {"role": "user", "content": "And tomorrow?"}], "tools": TOOLS}
        messages, tools, kwargs = openai_to_messages(req)
        out = self.t.render(messages, tools, **kwargs)
        self.assertIn(GEN + "\n<think>Call the tool.</think><tool_call>get_weather\n<arg_key>city</arg_key>\n"
                      "<arg_value>Paris</arg_value><arg_key>days</arg_key>\n<arg_value>2</arg_value>\n</tool_call>"
                      "<|role_end|>", out)
        self.assertIn("<role>OBSERVATION</role>\n<tool_response>\n{\"temp\": 21}\n</tool_response><|role_end|>", out)
        self.assertIn(GEN + "\n<think></think>21 degrees.<|role_end|><role>HUMAN</role>And tomorrow?<|role_end|>", out)

    def test_next_turn_prefix(self):
        """A turn's prompt is the start of the next one's whatever the answer was (the engine's snapshot then applies),
        and with the answer as the parser reports it, exactly the prompt plus the answer's text."""
        for kw in ({}, {"enable_thinking": False}):
            first = [{"role": "user", "content": "Name a colour."}]
            p1 = self.t.render(first, **kw)
            reasoning = "" if kw else "The user wants a colour."
            answer = "Blue."
            p2 = self.t.render(first + [{"role": "assistant", "content": answer, "reasoning_content": reasoning or None},
                                        {"role": "user", "content": "Another?"}], **kw)
            self.assertTrue(p2.startswith(p1), kw)
            written = (reasoning + "</think>" + answer) if reasoning else answer
            self.assertTrue(p2.startswith(p1 + written + "<|role_end|>"), (kw, p2[len(p1):len(p1) + 80]))


class Parse(unittest.TestCase):
    def test_reasoning_and_answer(self):
        for seed in range(5):
            r, c, calls = events("Let me think.\nOK.</think>Paris.", seed=seed)
            self.assertEqual((r.strip(), c, calls), ("Let me think.\nOK.", "Paris.", []))
        r, c, calls = events("Paris.", thinking=False)
        self.assertEqual((r, c), ("", "Paris."))

    def test_tool_call(self):
        out = ("I will check.</think>Checking.\n<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris"
               "</arg_value>\n<arg_key>days</arg_key>\n<arg_value>3</arg_value>\n</tool_call>")
        for seed in range(5):
            r, c, calls = events(out, seed=seed)
            self.assertEqual(r, "I will check.")
            self.assertEqual(len(calls), 1)
            self.assertEqual((calls[0].name, calls[0].arguments), ("get_weather", {"city": "Paris", "days": 3}))


if __name__ == "__main__":
    unittest.main()
