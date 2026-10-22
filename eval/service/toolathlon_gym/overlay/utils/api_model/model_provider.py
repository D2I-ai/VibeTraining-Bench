"""
CAMEL ModelFactory wrapper.
Supports env var overrides: MODEL_PLATFORM, MODEL_API_KEY, MODEL_API_URL
"""
import os
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from configs.global_configs import global_configs

# provider name → (ModelPlatformType, api_key_fn, default_base_url)
_PROVIDER_MAP = {
    "aihubmix":   (ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
                   lambda: global_configs.aihubmix_key,
                   "https://aihubmix.com/v1"),
    "openai":     (ModelPlatformType.OPENAI,
                   lambda: global_configs.openai_official_key, None),
    "anthropic":  (ModelPlatformType.ANTHROPIC,
                   lambda: global_configs.anthropic_official_key, None),
    "deepseek":   (ModelPlatformType.DEEPSEEK,
                   lambda: global_configs.deepseek_official_key, None),
    "openrouter": (ModelPlatformType.OPENROUTER,
                   lambda: global_configs.openrouter_key, None),
    "qwen":       (ModelPlatformType.QWEN,
                   lambda: global_configs.qwen_official_key, None),
    "gemini":     (ModelPlatformType.GEMINI,
                   lambda: global_configs.google_official_key, None),
    "openai_compatible": (ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
                          lambda: os.environ.get("MODEL_API_KEY", ""), None),
}


def build_model(model_name: str, provider: str):
    """Build a CAMEL BaseModelBackend.

    Env var overrides (take priority over provider map):
      MODEL_PLATFORM  - CAMEL ModelPlatformType name or provider key
      MODEL_API_KEY   - API key
      MODEL_API_URL   - base URL (for compatible endpoints)
    """
    # Env var overrides
    env_platform = os.environ.get("MODEL_PLATFORM")
    env_key      = os.environ.get("MODEL_API_KEY")
    env_url      = os.environ.get("MODEL_API_URL")

    if env_platform:
        # Try to resolve as provider key first, then as ModelPlatformType name
        if env_platform.lower() in _PROVIDER_MAP:
            platform, key_fn, default_url = _PROVIDER_MAP[env_platform.lower()]
        else:
            platform = ModelPlatformType[env_platform.upper()]
            key_fn = lambda: ""
            default_url = None
        api_key = env_key or key_fn()
        url = env_url or default_url
    else:
        if provider not in _PROVIDER_MAP:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Supported: {list(_PROVIDER_MAP.keys())}"
            )
        platform, key_fn, default_url = _PROVIDER_MAP[provider]
        api_key = env_key or key_fn()
        url = env_url or default_url

    kwargs = dict(model_platform=platform, model_type=model_name, api_key=api_key)
    if url:
        # Anthropic SDK uses base_url without /v1 suffix; others need /v1
        if platform == ModelPlatformType.ANTHROPIC:
            kwargs["url"] = url.rstrip("/").rstrip("/v1")
        else:
            kwargs["url"] = url.rstrip("/") + ("/v1" if not url.rstrip("/").endswith("/v1") else "")

    # The 48h r1 run (final job 0ef68ce2b287, 2026-09-06) lost 42/96 tasks to
    # agent_crash via httpcore.ReadTimeout: camel's per-attempt timeout falls
    # back to 180s when MODEL_TIMEOUT is unset, and requests carry no
    # max_tokens, so sglang generates up to the remaining context window
    # (~230k tokens) whenever the fine-tuned checkpoint loses its stop
    # behaviour mid-task -- that is hours of decode, not minutes. The sibling
    # evidence: 767 requests from the same crashed tasks finished in median
    # 1.3s, the timing-out requests simply never returned, and one legitimate
    # slow request took 404s. Three client-side bounds make a verdict
    # (pass/fail) reachable again instead of agent_crash:
    #   * max_tokens caps a run-away generation at a bounded wait; the
    #     benchmark declares the cap and healthy steps never approach it;
    #   * timeout=900s keeps the 404s-class slow-but-valid requests alive with
    #     headroom over the capped worst case;
    #   * max_retries=2 lets the OpenAI SDK re-issue a request that landed in
    #     a server-side black hole instead of failing the whole task.
    # Source of truth for these three is benchmarks/eigent-ai__toolathlon_gym/benchmark.json
    # (agent_kwargs): evaluate.py posts them and runner.py injects them here as
    # TOOLATHLON_* env. The literals below are only the fallback when unset.
    extra = dict(kwargs.get("model_config_dict") or {})
    extra.setdefault("max_tokens", int(os.environ.get("TOOLATHLON_MAX_TOKENS") or 8192))
    for key, caster in (
        ("temperature", float),
        ("top_p", float),
        ("top_k", int),
        ("repetition_penalty", float),
    ):
        value = os.environ.get(f"TOOLATHLON_{key.upper()}")
        if value is not None:
            extra[key] = caster(value)
    kwargs["model_config_dict"] = extra
    kwargs["timeout"] = float(os.environ.get("TOOLATHLON_MODEL_TIMEOUT") or 900)
    kwargs["max_retries"] = int(os.environ.get("TOOLATHLON_MODEL_RETRIES") or 2)

    # Qwen3.5's chat template pre-fills the '<think>' opener into the prompt unless
    # enable_thinking is explicitly false (see chat_template.jinja, the else-branch of
    # the `enable_thinking is defined and enable_thinking is false` test), so the model
    # only ever emits the closing '</think>'. vLLM's Qwen3ReasoningParser is written for
    # exactly that -- but when a reply is missing the closing tag it takes a degrade
    # path that hands back `(everything, None)`: the whole generation becomes reasoning
    # and content is None. The tool call parser only looks at content, so tool_calls
    # comes back empty too, and the agent sees an empty reply, ends the turn, and the
    # task reports "Ended without claim_done" after ~30 seconds. Confirmed against a
    # conv log from 08-19: reasoning held a well-formed <tool_call>, content was null,
    # tool_calls was [].
    #
    # Two things this is NOT, both of which were written here before and were wrong:
    # it is not a vLLM version problem (0.17.1 with --reasoning-parser qwen3 works;
    # upgrading to 0.27.1 broke the PPU driver instead and was rolled back), and the
    # parser does not assume paired tags. What triggers the missing '</think>' is still
    # unknown -- five reproduction attempts against a 0.17.1 endpoint (40-sample single
    # turn, 3..80 tools, 48-round replay, concurrency sweep, provider control) all came
    # back clean, so treat it as low-frequency rather than absent.
    #
    # Default follows the model's NATIVE template (thinking ON for Qwen3.5 -- how our
    # SFT checkpoints were trained, and what the official verified harness sends): we do
    # NOT inject enable_thinking unless explicitly opted out. Set TOOLATHLON_ENABLE_THINKING=0
    # for a base/non-thinking model -- the template then pre-fills a matched
    # '<think>\n\n</think>' and the parser takes its safe (content-populated) branch.
    # The service sets this env per container from benchmark.json
    # (agent_kwargs.llm_call_kwargs), a per-job switch, so base and SFT can
    # run concurrently. This matches toolathlon_verified/utils/api_model/model_provider.py.
    #
    # It goes under extra_body, NOT at the top level: chat_template_kwargs is a
    # vLLM extension rather than an OpenAI SDK argument, and the SDK rejects an
    # unknown top-level keyword outright -- "AsyncCompletions.create() got an
    # unexpected keyword argument 'chat_template_kwargs'", which reaches the
    # results as an agent crash on every task instead of as a config mistake.
    if os.environ.get("TOOLATHLON_ENABLE_THINKING", "").strip().lower() in ("0", "false", "no"):
        extra = dict(kwargs.get("model_config_dict") or {})
        body = dict(extra.get("extra_body") or {})
        chat_kwargs = dict(body.get("chat_template_kwargs") or {})
        chat_kwargs["enable_thinking"] = False
        body["chat_template_kwargs"] = chat_kwargs
        extra["extra_body"] = body
        kwargs["model_config_dict"] = extra

    return ModelFactory.create(**kwargs)
