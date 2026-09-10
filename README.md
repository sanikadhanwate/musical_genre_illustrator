---
title: Musical Genre Illustrator
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_scopes:
- inference-api
short_description: ' '
---

An example chatbot using [Gradio](https://gradio.app), [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/v0.22.2/en/index), and the [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index).

## Adaptive LLM failover (extra credit)

The prompt-writing step (`create_visual_prompt` in [app.py](app.py)) normally
calls a **remote** hosted LLM (`meta-llama/Llama-3.1-8B-Instruct` via the HF
Inference API). It can automatically fail over to a small **local** LLM
(`Qwen/Qwen2.5-0.5B-Instruct`, run on-device with `transformers`) when the
remote API is not usable — no manual model switch is required.

**a. Failure detection.** The remote call is wrapped in a `try/except` with a
15s client timeout. Any exception — connection error, timeout, HTTP 429 rate
limit, or missing login — is caught and classified (`timeout`, `rate
limited`, or `unavailable`) in `_classify_remote_failure`.

**b. Automatic routing.** On failure, `_RemoteLLMCircuitBreaker` marks the
remote LLM "down" for 60 seconds (`REMOTE_COOLDOWN_SECONDS`) and the very
same request is immediately re-sent to the local LLM — the user never sees
an error. While in cooldown, subsequent requests skip the remote call
entirely and go straight to local, so a known-bad API isn't retried on every
click. Once the cooldown elapses, the next request automatically tries
remote again, so the app "heals" back without any user action. If the local
LLM also fails, the app falls back one more level to a static per-genre
prompt template so the pipeline never dead-ends.

**c. Indicating which model handled the request.** `create_visual_prompt`
returns `(prompt_text, source_label)`; the Gradio UI shows the label in the
"Prompt written by" field, e.g. `Remote LLM (meta-llama/Llama-3.1-8B-Instruct)`
or `Local LLM (Qwen/Qwen2.5-0.5B-Instruct) — automatic failover`.

**d. Demonstrating the behavior.** Since a real outage or rate limit is hard
to trigger on demand, the UI has a **"Simulate remote LLM outage"**
checkbox that forces the remote call to raise before it's attempted. Check
it and click "Analyze Music" — the "Prompt written by" field will show the
local LLM handled the request, and the console prints the
`[FAILOVER]` log lines. Automated tests for this logic (stubbing out the
local model download) live in [test_app.py](test_app.py):
`test_simulated_remote_failure_routes_to_local_llm`,
`test_remote_stays_in_cooldown_after_a_failure_and_skips_straight_to_local`,
and `test_both_llms_unavailable_falls_back_to_static_template`.

**e. Advantages and disadvantages.**

- *Advantages*: the app stays available during remote outages, rate limits,
  or network issues, with no user-visible errors and no manual switching;
  the cooldown avoids hammering a struggling API; degrading further to a
  static template guarantees the pipeline always returns *something*.
- *Disadvantages*: the local model is much smaller (0.5B vs 8B parameters),
  so failover prompts are lower quality than the remote model's; loading it
  on first use adds latency and memory/GPU usage to whichever request
  triggers the failover; the cooldown window is a blunt instrument — a
  transient blip causes 60s of unnecessarily using the (weaker) local model,
  while a longer outage might need a longer cooldown to avoid repeated
  failed remote attempts; and silently routing around failures can mask a
  systemic problem (e.g. an expired token) that would be better surfaced to
  the user.
