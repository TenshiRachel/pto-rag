import time
from langchain.callbacks.base import BaseCallbackHandler

class TimingCallback(BaseCallbackHandler):
    """
    Tracks:
    - Tool timings (per tool)
    - Token usage per LLM call
    - Consolidated LLM reasoning time
    - Total wall time
    - Cache hits (we register manually from main.py)
    """

    def __init__(self):
        # ----- overall session timing -----
        self.session_start = None
        self.session_end = None

        # ----- tool timing -----
        self.tool_timings = []          # list[{"tool": str, "time": float}]
        self.current_tool_name = None
        self.current_tool_start_time = None
        self.last_tool_end_time = None  # timestamp of last finished tool
        self.total_tool_time = 0.0

        # we'll derive retrieve time later by summing times of tools with "retriev" in their name

        # ----- LLM timing -----
        self.llm_start_time = None
        self.reasoning_total = 0.0      # sum of LLM call durations (planning/thinking)
        self.generation_total = 0.0     # time after last tool until final answer stream
        self.llm_call_count = 0

        # contextual label for each LLM call
        self.llm_context = None

        # ----- token usage -----
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.llm_calls = []  # per-call token breakdown

        # ----- cache stats -----
        self.cache_hits = 0
        self.cache_misses = 0

    # -----------------------
    # MANUAL SESSION START/END
    # -----------------------
    def start_total_timer(self):
        self.session_start = time.perf_counter()

    def end_total_timer(self):
        self.session_end = time.perf_counter()
        # if we had at least one tool call, estimate generation_total
        if self.last_tool_end_time is not None:
            self.generation_total = self.session_end - self.last_tool_end_time

    # -----------------------
    # TOOL TIMING
    # -----------------------
    def on_tool_start(self, serialized, input_str, **kwargs):
        self.current_tool_name = serialized.get("name", "UnknownTool")
        self.current_tool_start_time = time.perf_counter()
        # (optional) debug print
        print(f"[Tool start] {self.current_tool_name}")

    def on_tool_end(self, output, **kwargs):
        if self.current_tool_start_time is None:
            return  # should not happen, but guard anyway

        end_time = time.perf_counter()
        duration = end_time - self.current_tool_start_time

        # record timing
        self.tool_timings.append({
            "tool": self.current_tool_name,
            "time": duration
        })
        self.total_tool_time += duration

        # mark last tool end (used later to compute generation_total)
        self.last_tool_end_time = end_time

        # reset current tool
        print(f"[Tool end] {self.current_tool_name} took {duration:.3f}s")
        self.current_tool_name = None
        self.current_tool_start_time = None

    def register_cache_hit(self, hit: bool):
        """Called from main.py after agent.invoke()."""
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    # -----------------------
    # LLM TIMING / TOKENS
    # -----------------------
    def on_llm_start(self, *args, **kwargs):
        self.llm_call_count += 1
        self.llm_start_time = time.perf_counter()

        # tag the context of this call
        if self.current_tool_name:
            # LLM is being called 'inside' a tool loop, rare, but keep it
            self.llm_context = f"during_{self.current_tool_name}"
        else:
            if self.llm_call_count == 1:
                self.llm_context = "initial_planning"
            else:
                self.llm_context = "reasoning"

    def on_llm_end(self, response, **kwargs):
        # accumulate reasoning time
        if self.llm_start_time is not None:
            duration = time.perf_counter() - self.llm_start_time
            self.reasoning_total += duration
            print(f"[LLM call #{self.llm_call_count}] {duration:.3f}s ({self.llm_context})")
            self.llm_start_time = None

        # extract token usage
        try:
            usage = None

            # (a) try generations[].message.usage_metadata (works with ChatOpenAI in langchain_openai)
            if hasattr(response, "generations"):
                for gen_list in response.generations:
                    if isinstance(gen_list, list):
                        for gen in gen_list:
                            if hasattr(gen, "message") and hasattr(gen.message, "usage_metadata"):
                                usage = gen.message.usage_metadata
                                break
                    if usage:
                        break

            if usage:
                # handle dict-like or attr-like
                if isinstance(usage, dict):
                    prompt = usage.get("prompt_tokens") or usage.get("input_tokens", 0)
                    completion = usage.get("completion_tokens") or usage.get("output_tokens", 0)
                    total = usage.get("total_tokens", 0)
                else:
                    prompt = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0)
                    completion = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0)
                    total = getattr(usage, "total_tokens", 0)

                self.prompt_tokens += prompt
                self.completion_tokens += completion
                self.total_tokens += total

                self.llm_calls.append({
                    "call_number": self.llm_call_count,
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "total_tokens": total,
                    "context": self.llm_context or "unknown",
                    "timestamp": (time.perf_counter() - self.session_start) if self.session_start else 0.0
                })

            else:
                print("[warn] No token usage metadata found in llm response")

        except Exception as e:
            print(f"[error extracting tokens] {e}")

        # clear context for next call
        self.llm_context = None

    # -----------------------
    # SUMMARY
    # -----------------------
    def get_summary(self):
        # compute total session time
        if self.session_end is None and self.session_start is not None:
            total_elapsed = time.perf_counter() - self.session_start
        elif self.session_end is not None and self.session_start is not None:
            total_elapsed = self.session_end - self.session_start
        else:
            total_elapsed = None

        # aggregate tool times
        t_tool_total = sum(t["time"] for t in self.tool_timings)
        t_retrieve = sum(
            t["time"]
            for t in self.tool_timings
            if "retriev" in (t["tool"] or "").lower()
        )

        return {
            "T_retrieve": round(t_retrieve, 4),
            "T_tools": [
                {"tool": rec["tool"], "time": round(rec["time"], 4)}
                for rec in self.tool_timings
            ],
            "T_tools_total": round(t_tool_total, 4),
            "T_reason": round(self.reasoning_total, 4),
            "T_generate": round(self.generation_total, 4),
            "T_total": round(total_elapsed, 4) if total_elapsed is not None else None,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
            "num_llm_calls": self.llm_call_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }

    def print_token_breakdown(self):
        """Optional pretty printer if you want to dump token usage."""
        print("\n=== Token Usage Breakdown ===")
        print(f"{'Call':<6} {'Prompt':<8} {'Completion':<12} {'Total':<8} {'Context'}")
        print("-" * 70)
        for call in self.llm_calls:
            print(
                f"#{call['call_number']:<5} "
                f"{call['prompt_tokens']:<8} "
                f"{call['completion_tokens']:<12} "
                f"{call['total_tokens']:<8} "
                f"{call['context']}"
            )
        print("-" * 70)
        print(
            f"{'TOTAL':<6} "
            f"{self.prompt_tokens:<8} "
            f"{self.completion_tokens:<12} "
            f"{self.total_tokens:<8}"
        )
