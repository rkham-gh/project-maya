"""serve/test_inkling_frontend.py - Inkling-Small in the frontend: its chat template (tools/ink/chat_template.jinja, the
one its GGUF carries) rendered with the server's thinking levels and tool results, and the model's typed content blocks
(reasoning, answer, tool calls) parsed from any split of the stream.

    python -m unittest serve.test_inkling_frontend
"""
from __future__ import annotations

import json
import random
import unittest
from pathlib import Path

from serve.frontend import (ChatTemplate, OutputParser, effort_kwargs, ink_tool_names, openai_to_messages)

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "tools" / "ink" / "chat_template.jinja"
M, TH, TX, END, TOOL, STOP = ("<|message_model|>", "<|content_thinking|>", "<|content_text|>", "<|end_message|>",
                              "<|content_invoke_tool_json|>", "<|content_model_end_sampling|>")
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "The weather of a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                   "required": ["city"]}}}]


def events(out: str, thinking=True, seed=0, prefill=""):
    """The parser's events for `out` fed in random pieces: (reasoning, content, tool calls)."""
    rng = random.Random(seed)
    p = OutputParser(thinking=thinking, tools=[t["function"] for t in TOOLS], stream_tools=True, inkling=True)
    evs = p.feed(prefill) if prefill else []
    i = 0
    while i < len(out):
        n = rng.randint(1, 9)
        evs += p.feed(out[i:i + n])
        i += n
    evs += p.finish()
    reasoning = "".join(e.text for e in evs if e.kind == "reasoning")
    content = "".join(e.text for e in evs if e.kind == "content")
    calls = [e.call for e in evs if e.kind == "tool_call"]
    return reasoning, content, calls


def call_of(c):
    args = c.arguments if hasattr(c, "arguments") else c["function"]["arguments"]
    name = c.name if hasattr(c, "name") else c["function"]["name"]
    return name, json.loads(args) if isinstance(args, str) else args


class Template(unittest.TestCase):
    def setUp(self):
        self.t = ChatTemplate(TEMPLATE)

    def test_family(self):
        self.assertTrue(self.t.inkling)
        self.assertFalse(self.t.minimax or self.t.deepseek or self.t.glm_effort)

    def test_levels(self):
        msgs = [{"role": "user", "content": "Hi"}]
        for effort, text in (("max", "0.99"), ("xhigh", "0.99"), ("high", "0.9"), ("medium", "0.7"), ("low", "0.2"),
                             ("none", "0"), (False, "0")):
            out = self.t.render(msgs, **effort_kwargs(effort))
            self.assertIn(f"<|message_system|><|content_text|>Thinking effort level: {text}{END}", out, effort)
        self.assertIn("Thinking effort level: 0.9" + END, self.t.render(msgs))        # (the template's default)
        self.assertTrue(self.t.render(msgs).endswith(M))

    def test_history(self):
        req = {"messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "reasoning_content": "Call the tool.", "content": "",
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\", \"days\": 2}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "{\"temp\": 21}"},
            {"role": "assistant", "content": "21 degrees."},
            {"role": "user", "content": "And tomorrow?"}], "tools": TOOLS}
        messages, tools, kwargs = openai_to_messages(req)
        out = self.t.render(messages, tools, **kwargs)
        self.assertIn(M + TH + "Call the tool." + END, out)
        self.assertIn(M + "get_weather" + TOOL + '{"name":"get_weather","args":{"city":"Paris","days":2}}' + END + STOP,
                      out)
        self.assertIn("<|message_tool|>get_weather" + TX + '{"temp": 21}' + END, out)
        self.assertIn(M + TX + "21 degrees." + END + STOP, out)
        self.assertIn("<|message_system|>tool_declare<|content_xml|>[", out)
        self.assertTrue(out.endswith("<|message_user|>" + TX + "And tomorrow?" + END + M))

    def test_tool_names(self):
        msgs = [{"role": "assistant", "content": "", "tool_calls": [
                    {"function": {"name": "a", "arguments": {}}}, {"function": {"name": "b", "arguments": {}}}]},
                {"role": "tool", "content": "1"}, {"role": "tool", "content": "2"},
                {"role": "tool", "content": "3", "name": "c"}]
        self.assertEqual([m.get("name") for m in ink_tool_names(msgs)], [None, "a", "b", "c"])


class Parser(unittest.TestCase):
    def test_think_then_answer(self):
        out = TH + "Let me <think> it.\n" + END + M + TX + "Hello <b>there</b>!" + END + STOP
        for seed in range(30):
            self.assertEqual(events(out, seed=seed)[:2], ("Let me <think> it.\n", "Hello <b>there</b>!"), seed)

    def test_answer_only(self):
        out = TX + "Just this." + END + STOP
        for seed in range(10):
            self.assertEqual(events(out, seed=seed), ("", "Just this.", []))
            self.assertEqual(events(out, thinking=False, seed=seed), ("", "Just this.", []))

    def test_thinks_when_off(self):
        out = TH + "hmm" + END + M + TX + "ok" + END + STOP
        for seed in range(10):
            r, c, _ = events(out, thinking=False, seed=seed)
            self.assertEqual((r, c), ("hmm", "ok"))

    def test_tool_call(self):
        out = (TH + "Need the weather." + END + M + "get_weather" + TOOL +
               '{"name":"get_weather","args":{"city":"Paris <Île>","days":3}}' + END + STOP)
        for seed in range(30):
            r, c, calls = events(out, seed=seed)
            self.assertEqual((r, c.strip()), ("Need the weather.", ""), seed)
            self.assertEqual([call_of(x) for x in calls], [("get_weather", {"city": "Paris <Île>", "days": 3})], seed)

    def test_text_then_two_calls(self):
        out = (TX + "Checking both." + END + M + "get_weather" + TOOL + '{"name":"get_weather","args":{"city":"Rome"}}' +
               END + M + "get_weather" + TOOL + '{"name":"get_weather","args":{"city":"Oslo","days":1}}' + END + STOP)
        for seed in range(30):
            r, c, calls = events(out, seed=seed)
            self.assertEqual(c.strip(), "Checking both.", seed)
            self.assertEqual([call_of(x) for x in calls], [("get_weather", {"city": "Rome"}),
                                                           ("get_weather", {"city": "Oslo", "days": 1})], seed)

    def test_forced_call(self):
        prefill = "get_weather" + TOOL + '{"name":"get_weather","args":'
        for seed in range(10):
            r, c, calls = events('{"city":"Lima"}}' + END + STOP, thinking=False, seed=seed, prefill=prefill)
            self.assertEqual([call_of(x) for x in calls], [("get_weather", {"city": "Lima"})])

    def test_cut_short(self):
        # max_tokens inside the thinking, inside the answer, inside a call: what came is kept, nothing invented
        self.assertEqual(events(TH + "half a thou")[:2], ("half a thou", ""))
        self.assertEqual(events(TH + "t" + END + M + TX + "half an ans")[:2], ("t", "half an ans"))
        r, c, calls = events(TH + "t" + END + M + "get_weather" + TOOL + '{"name":"get_weather","args":{"ci')
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
