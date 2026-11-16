"""Claude models through the Anthropic API, behind the same interface as agent.Agent (agent.py --model claude-...).

Same system prompt, tools and loop as the vLLM agent, so runs are comparable: ask the model, run the tools it calls
(through agent.call_tool, with the same output limit), repeat until it answers without a tool call, at most
max_turns calls per send(). The native Messages API (no OpenAI-compatible shim):
- thinking is adaptive (on Claude Opus 5.5 it is always on; there is no way to turn it off), depth set with effort;
- the conversation is cached (top-level cache_control), so each call pays full price only for what is new;
- thinking blocks are passed back unchanged, as the API requires;
- calls are streamed, so a long reply can't hit the HTTP timeout.
Each call's cost is computed from its usage, and a shared Budget refuses calls once --max-dollars is spent.
If the model declines a request (stop_reason "refusal"), the API can re-run it on another model (fallbacks), which
is counted per task ("fallbacks"), so those tasks can be set apart.

The API key comes from ANTHROPIC_API_KEY or, if that is unset, the file ~/.config/anthropic/api_key (chmod 600).
Never commit it.
"""
import json
import os
import threading
import time
from pathlib import Path

import anthropic

from agent import BOLD, CYAN, DIM, GREEN, YELLOW, _block, _c, _show_args, call_tool, new_stats

# $ per million tokens: input, output, cache read. A cache write (5-minute TTL) costs 1.25x input.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10),
    "claude-fable-5-2": (8.00, 40.00, 0.20),
    "claude-opus-5": (5.00, 25.00, 0.50),
    "claude-opus-4-8": (5.00, 25.00, 0.50),
    "claude-sonnet-5": (2.00, 10.00, 0.20),
}
CACHE_WRITE = 1.25
MAX_TOKENS = 64000          # per reply; streamed, so no timeout
FALLBACK_BETA = "server-side-fallback-2026-07-01"
KEY_FILE = Path.home() / ".config" / "anthropic" / "api_key"


class BudgetExceeded(Exception):
    """The run has spent its --max-dollars."""


class Budget:
    """Dollars spent by every agent of a run, checked before each call (a call in flight can go over by its own cost)."""

    def __init__(self, limit):
        self.limit, self.spent, self.lock = float(limit), 0.0, threading.Lock()

    def check(self):
        with self.lock:
            if self.spent >= self.limit:
                raise BudgetExceeded(f"spent ${self.spent:.2f} of the ${self.limit:.2f} limit (--max-dollars)")

    def add(self, dollars):
        with self.lock:
            self.spent += dollars

    @property
    def exhausted(self):
        with self.lock:
            return self.spent >= self.limit


def make_client():
    key = os.environ.get("ANTHROPIC_API_KEY") or (KEY_FILE.read_text().strip() if KEY_FILE.exists() else None)
    if not key:
        raise SystemExit(f"No Claude API key: set ANTHROPIC_API_KEY, or put the key in {KEY_FILE} (chmod 600).")
    return anthropic.Anthropic(api_key=key, max_retries=5)


def cost(model, usage):
    """Dollars for one call, from its usage."""
    p_in, p_out, p_read = PRICES[model]
    return ((usage.input_tokens or 0) * p_in + (getattr(usage, "cache_creation_input_tokens", 0) or 0) * p_in * CACHE_WRITE
            + (getattr(usage, "cache_read_input_tokens", 0) or 0) * p_read + (usage.output_tokens or 0) * p_out) / 1e6


def _dump(block):
    return block.model_dump(exclude_none=True) if hasattr(block, "model_dump") else block


class ClaudeAgent:
    """One reasoning loop on a Claude model; the same methods and stats as agent.Agent (run, send, stats, limit_hits)."""

    def __init__(self, client, model, budget, tools, system_prompt, max_turns=12, effort="high", verbose=True,
                 name="", max_tool_output=None, fallbacks=True, **_ignored):
        if model not in PRICES:
            raise SystemExit(f"No prices for {model}: add it to claude_agent.PRICES so its cost can be counted.")
        self.client, self.model, self.budget = client, model, budget
        self.tools, self.system_prompt, self.max_turns = tools, system_prompt, max_turns
        self.effort, self.verbose, self.name = effort, verbose, name
        self.max_tool_output, self.fallbacks = max_tool_output, fallbacks
        # The vLLM agent's tool schemas, in the Messages API's form.
        self.tool_defs = [{"name": s["function"]["name"], "description": s["function"]["description"],
                           "input_schema": s["function"]["parameters"]} for _, s in tools.values()]
        self.messages = []
        self.stats = self._new_stats()
        self.limit_hits = 0
        self.inbox = None  # as in agent.Agent: text that arrived while working (teammates), or None
        self._stats_lock = threading.Lock()

    @staticmethod
    def _new_stats():
        return {**new_stats(), "dollars": 0.0, "cache_read_tokens": 0, "cache_write_tokens": 0, "fallbacks": 0,
                "refusals": 0}

    def log(self, text):
        if self.verbose:
            print(text, flush=True)

    def transcript(self):
        """The conversation as plain JSON (content blocks dumped), for the run's traces."""
        return [{"role": "system", "content": self.system_prompt}] + [
            {"role": m["role"], "content": m["content"] if isinstance(m["content"], str) else [_dump(b) for b in m["content"]]}
            for m in self.messages]

    def _result(self, final, stop):
        return {"final": final, "stop": stop, "messages": self.transcript(), **self.stats}

    def run(self, task):
        self.messages, self.stats, self.limit_hits = [], self._new_stats(), 0
        return self.send(task)

    def send(self, text):
        self.messages.append({"role": "user", "content": text})
        for _ in range(self.max_turns):
            resp = self._call()
            self.messages.append({"role": "assistant", "content": resp.content})  # thinking blocks passed back as is
            said = "".join(b.text for b in resp.content if b.type == "text")
            uses = [b for b in resp.content if b.type == "tool_use"]
            if resp.stop_reason == "refusal":
                with self._stats_lock:
                    self.stats["refusals"] += 1
                self.log(_c(YELLOW, f"  {self.name or 'agent'}: the model declined ({resp.stop_details})"))
                return self._result(said, "refusal")
            if not uses:
                self.log(_block("answer", said, GREEN))
                return self._result(said, "final" if resp.stop_reason != "max_tokens" else "max_tokens")
            if said:
                self.log(_block("says", said, DIM))
            results = []
            for b in uses:
                if resp.stop_reason == "max_tokens":  # cut off mid-call: the input may be incomplete, so don't run it
                    out = "[error] your reply hit the output limit before this call was complete; nothing ran. Call again."
                else:
                    self.log(_block(b.name, _show_args(json.dumps(b.input)), CYAN, limit=3000))
                    out = call_tool(b.name, json.dumps(b.input), self.tools, max_output=self.max_tool_output)
                failed = out.startswith("[error]") or "[exit code" in out
                with self._stats_lock:
                    self.stats["tool_calls"] += 1
                    self.stats["tool_errors"] += failed
                self.log(_block("output", out, YELLOW if failed else DIM, limit=800))
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": out,
                                **({"is_error": True} if out.startswith("[error]") else {})})
            self.messages.append({"role": "user", "content": results})
            arrived = self.inbox() if self.inbox else None
            if arrived:
                self.log(_block("arrived", arrived, DIM, limit=800))
                self.messages.append({"role": "user", "content": arrived})
        self.limit_hits += 1
        self.log(_c(YELLOW, f"  {self.name or 'agent'} stopped: hit --max-turns ({self.max_turns}) without a final answer"))
        return self._result("", "max_turns")

    def _call(self):
        self.budget.check()
        started = time.time()
        params = dict(model=self.model, max_tokens=MAX_TOKENS, system=self.system_prompt, tools=self.tool_defs,
                      messages=self.messages, output_config={"effort": self.effort},
                      cache_control={"type": "ephemeral"})  # cache the conversation up to its last block
        if self.fallbacks:
            with self.client.beta.messages.stream(betas=[FALLBACK_BETA], fallbacks="default", **params) as stream:
                resp = stream.get_final_message()
        else:
            with self.client.messages.stream(**params) as stream:
                resp = stream.get_final_message()
        u = resp.usage
        dollars = cost(self.model, u)
        self.budget.add(dollars)
        fell_back = sum(getattr(i, "type", None) == "fallback_message" for i in (getattr(u, "iterations", None) or []))
        with self._stats_lock:
            self.stats["turns"] += 1
            turn = self.stats["turns"]
            read, write = getattr(u, "cache_read_input_tokens", 0) or 0, getattr(u, "cache_creation_input_tokens", 0) or 0
            self.stats["prompt_tokens"] += (u.input_tokens or 0) + read + write
            self.stats["completion_tokens"] += u.output_tokens or 0
            self.stats["cache_read_tokens"] += read
            self.stats["cache_write_tokens"] += write
            self.stats["dollars"] = round(self.stats["dollars"] + dollars, 6)
            self.stats["fallbacks"] += fell_back
        who = f"{self.name} " if self.name else ""
        self.log(_c(BOLD, f"── {who}turn {turn} ──") + _c(DIM, f" {u.input_tokens} new + {read} cached + {write} cache-write "
                                                             f"in, {u.output_tokens} out, ${dollars:.3f} (run ${self.budget.spent:.2f}), "
                                                             f"{time.time() - started:.1f}s" + (" FALLBACK" if fell_back else "")))
        return resp
