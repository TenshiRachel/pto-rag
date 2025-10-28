import time
from langchain.callbacks.base import BaseCallbackHandler


class TimingCallback(BaseCallbackHandler):
    """
    Tracks:
    - Tool timings (per tool)
    - Token usage per LLM call
    - Consolidated LLM reasoning time
    - Total session time
    """

    def __init__(self):
        self.tool_timings = []
        self.current_tool = None
        self.tool_start_time = None

        self.llm_start_time = None
        self.reasoning_total = 0.0
        self.generate_total = 0.0

        # NEW: Track when last tool ended
        self.last_tool_end_time = None
        self.generation_start_time = None

        # --- Token counts ---
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

        # NEW: Track tokens per LLM call
        self.llm_calls = []
        self.llm_call_count = 0

        # NEW: Track context for current LLM call
        self.llm_context = None  # Will be set in on_llm_start

        self.session_start = time.perf_counter()
        self.session_end = None

    # -----------------------
    # TOOL TIMING
    # -----------------------
    def on_tool_start(self, serialized, input_str, **kwargs):
        self.current_tool = serialized.get("name", "UnknownTool")
        self.tool_start_time = time.perf_counter()
        print(f"[Tool start] {self.current_tool}")

    def on_tool_end(self, output, **kwargs):
        if self.tool_start_time:
            duration = time.perf_counter() - self.tool_start_time
            self.tool_timings.append((self.current_tool, duration))
            print(f"[Tool end] {self.current_tool} ({duration:.3f}s)")
            # Track when the last tool ended
            self.last_tool_end_time = time.perf_counter()
        self.current_tool = None
        self.tool_start_time = None

    # -----------------------
    # LLM REASONING TIMING
    # -----------------------
    def on_llm_start(self, *args, **kwargs):
        self.llm_start_time = time.perf_counter()
        self.llm_call_count += 1

        # Capture the context at START time
        if self.current_tool:
            self.llm_context = f"during_{self.current_tool}"
        else:
            # Determine purpose based on call number
            if self.llm_call_count == 1:
                self.llm_context = "initial_planning"
            else:
                self.llm_context = "reasoning"

    def on_llm_end(self, response, **kwargs):
        # Timing
        if self.llm_start_time:
            duration = time.perf_counter() - self.llm_start_time
            self.reasoning_total += duration
            print(f"[LLM reasoning segment] {duration:.3f}s")
        self.llm_start_time = None

        # Token usage extraction
        try:
            usage = None

            # Method 1: Check llm_output
            # if hasattr(response, 'llm_output') and response.llm_output:
                # usage = response.llm_output.get('token_usage', None)

            # Method 2: Check usage_metadata attribute directly
            # if not usage and hasattr(response, 'usage_metadata'):
            #     usage = response.usage_metadata

            # Method 3: Check generations -> message -> usage_metadata
            if not usage and hasattr(response, 'generations'):
                for gen_list in response.generations:
                    if isinstance(gen_list, list):
                        for gen in gen_list:
                            if hasattr(gen, 'message') and hasattr(gen.message, 'usage_metadata'):
                                usage = gen.message.usage_metadata
                                break
                    if usage:
                        break

            # Extract tokens if we found usage data
            if usage:
                # Handle both dict and object formats
                if isinstance(usage, dict):
                    prompt = usage.get("prompt_tokens") or usage.get("input_tokens", 0)
                    completion = usage.get("completion_tokens") or usage.get("output_tokens", 0)
                    total = usage.get("total_tokens", 0)
                else:
                    prompt = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0)
                    completion = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0)
                    total = getattr(usage, "total_tokens", 0)

                # Add to totals
                self.prompt_tokens += prompt
                self.completion_tokens += completion
                self.total_tokens += total

                # Track this individual LLM call with captured context
                call_info = {
                    "call_number": self.llm_call_count,
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "total_tokens": total,
                    "context": self.llm_context or "unknown",  # Use captured context
                    "timestamp": time.perf_counter() - self.session_start
                }
                self.llm_calls.append(call_info)

                print(f"[Tokens Call #{self.llm_call_count} ({self.llm_context})] Prompt: {prompt}, Completion: {completion}, Total: {total}")

                # Clear context for next call
                self.llm_context = None
            else:
                print("[Warning] No token usage found in LLM response")

        except Exception as e:
            print(f"[Error extracting tokens] {e}")
            import traceback
            traceback.print_exc()

    # -----------------------
    # GENERATION (time from last tool to end)
    # -----------------------
    def on_llm_new_token(self, token, **kwargs):
        """Track when generation starts (first token after last tool)"""
        if self.last_tool_end_time and self.generation_start_time is None:
            self.generation_start_time = time.perf_counter()

    def finalize(self):
        # Calculate generation time as: time from last tool end to session end
        if self.last_tool_end_time:
            self.generate_total = time.perf_counter() - self.last_tool_end_time
        self.session_end = time.perf_counter()

    # -----------------------
    # SUMMARY
    # -----------------------
    def get_summary(self):
        t_tool_total = sum(t for _, t in self.tool_timings)
        t_retrieve = sum(
            t for name, t in self.tool_timings if "retriev" in name.lower()
        )

        return {
            "T_retrieve": round(t_retrieve, 4),
            "T_tools": [
                {"tool": n, "time": round(t, 4)} for n, t in self.tool_timings
            ],
            "T_tools_total": round(t_tool_total, 4),
            "T_reason": round(self.reasoning_total, 4),
            "T_generate": round(self.generate_total, 4),
            "T_total": round(
                (self.session_end or time.perf_counter()) - self.session_start, 4
            ),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
            "num_llm_calls": self.llm_call_count
        }

    def print_token_breakdown(self):
        """Print a detailed breakdown of token usage per LLM call"""
        print("\n=== Token Usage Breakdown ===")
        print(f"{'Call':<6} {'Prompt':<8} {'Completion':<12} {'Total':<8} {'Context'}")
        print("-" * 70)
        for call in self.llm_calls:
            print(f"#{call['call_number']:<5} {call['prompt_tokens']:<8} {call['completion_tokens']:<12} {call['total_tokens']:<8} {call['context']}")
        print("-" * 70)
        print(f"{'TOTAL':<6} {self.prompt_tokens:<8} {self.completion_tokens:<12} {self.total_tokens:<8}")
