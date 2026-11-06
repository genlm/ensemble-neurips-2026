"""Model loading for token- and byte-level ensembles.

Each function's docstring says what breaks without it.
"""

import logging

import torch
from genlm.bytes import BeamParams
from genlm.control import ByteLLM, PromptedLLM

logger = logging.getLogger(__name__)

# The paper's byte-beam settings (genlm-bytes defaults differ, and K changes the
# approximation, so they are pinned).
BEAM_K = 20
PRUNE_THRESHOLD = 0.005

QWEN_BOS = "<|im_start|>"


def qwen_bos(tokenizer):
    """Qwen tokenizers have no BOS; the paper's runners used ``<|im_start|>`` in its place."""
    if "qwen" in (getattr(tokenizer, "name_or_path", "") or "").lower():
        tokenizer.bos_token = QWEN_BOS
        return QWEN_BOS
    return None


def strip_bos(prompt_str: str, bos: str | None) -> str:
    """Byte-level prompt: the chat-formatted string with the leading BOS marker removed.

    The byte beam tokenises the prompt from raw bytes, and a special BOS token cannot be
    produced from bytes, so the paper's byte runners dropped it (everything up to the
    first ``<|begin_of_text|>`` for Llama, up to the first ``<|im_start|>`` for Qwen).
    """
    if bos is not None:
        return bos.join(prompt_str.split(bos)[1:])
    return prompt_str.encode("utf-8").split(b"<|begin_of_text|>")[-1].decode("utf-8")


def clamp_logprobs_to_vocab(llm):
    """Trim ``next_token_logprobs`` to the tokenizer's real vocabulary. Idempotent.

    Qwen2.5 pads its output layer: 152064 logits for 151665 tokens. genlm-bytes' trie
    asserts the two agree and dies with ``[152064, 151665]``; the padding sits at the top
    ids, so truncating is the correct projection.
    """
    if getattr(llm, "_logprobs_clamped", False):
        return
    n = len(llm.byte_vocab)
    orig = llm.next_token_logprobs

    async def clamped(context, **kwargs):
        lp = await orig(context, **kwargs)
        if getattr(lp, "shape", (n,))[0] != n:
            lp = lp[:n]
        return lp

    llm.next_token_logprobs = clamped
    llm._logprobs_clamped = True


def _card_gb():
    return torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0


def gpus_needed(model_name):
    """phi-4 (14B, bf16) does not fit a 24 GB card: shard it over two there."""
    return 2 if "phi" in model_name.lower() and _card_gb() < 40 else 1


def _layout(model_1, model_2):
    """GPU indices for each model of an HF pair: consecutive cards, phi-4 taking two on 24 GB cards.
    With a single visible card of >= 80 GB both models share it."""
    if torch.cuda.device_count() == 1 and _card_gb() >= 80:
        return [0], [0]
    n1, n2 = gpus_needed(model_1), gpus_needed(model_2)
    g1 = list(range(n1))
    g2 = g1 if model_2 == model_1 else list(range(n1, n1 + n2))
    need = max(g1 + g2) + 1
    if torch.cuda.device_count() < need:
        raise RuntimeError(f"{model_1} x {model_2} needs {need} GPUs, {torch.cuda.device_count()} visible")
    return g1, g2


def _hf_opts(gpus):
    """One card: pin to it. Several: shard with accelerate, 21 GiB per 24 GB card."""
    if len(gpus) == 1:
        return {"torch_dtype": torch.bfloat16, "device_map": {"": f"cuda:{gpus[0]}"}}
    return {"torch_dtype": torch.bfloat16, "device_map": "auto", "max_memory": {g: "21GiB" for g in gpus}}


def load_token_llms(model_1, model_2, backend="vllm", max_model_len=2048,
                    gpu_memory_utilization=0.95, temperature=1.0, tensor_parallel=1):
    """Two ``PromptedLLM``s for token-level ensembling.

    * ``vllm`` (the paper's setting) needs ``engine_opts`` passed explicitly -- ``from_name``
      silently drops unknown kwargs, and without ``max_model_len`` the engine
      builds at the model's full context and dies on KV-cache size. Only one engine can
      live in a process (genlm-backend forces ``VLLM_ENABLE_V1_MULTIPROCESSING=0``), so a
      cross-model pair must use ``hf``.
    * ``hf`` puts one model per visible GPU (phi-4 sharded over two, see ``_layout``).
    """
    if backend == "vllm":
        if model_1 != model_2:
            raise ValueError("two vLLM engines cannot share a process; use --backend hf for a cross-model pair")
        engine_opts = {
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "dtype": "half",
            "enable_prefix_caching": True,
        }
        if "phi" in model_1.lower():
            engine_opts["disable_sliding_window"] = True
        if tensor_parallel > 1:  # phi-4 over two 24 GB cards
            engine_opts.update(tensor_parallel_size=tensor_parallel, enforce_eager=True)
        llm1 = PromptedLLM.from_name(model_1, backend="vllm", engine_opts=engine_opts, temperature=temperature)
        return llm1, llm1.spawn()
    if backend == "hf":
        g1, g2 = _layout(model_1, model_2)
        llm1 = PromptedLLM.from_name(model_1, backend="hf", hf_opts=_hf_opts(g1), temperature=temperature)
        clamp_logprobs_to_vocab(llm1.model)
        if model_2 == model_1:
            return llm1, llm1.spawn()
        llm2 = PromptedLLM.from_name(model_2, backend="hf", hf_opts=_hf_opts(g2), temperature=temperature)
        clamp_logprobs_to_vocab(llm2.model)
        return llm1, llm2
    raise ValueError(f"unknown backend {backend!r}")


def _eos_bytes(llm) -> bytes:
    entry = llm.byte_vocab[llm.tokenizer.eos_token_id]
    return bytes(getattr(entry, "byte_string", entry))


def cache_byte_tries():
    """Build each model's token->byte trie once and reuse it for every beam.

    FLAG (library): ``ByteBeamState.initial`` rebuilds the trie from the vocabulary (~7 s for an
    8B vocab) for every new beam, and ``ByteLLM.set_prompt_from_str`` discards the initial beam,
    so each example starts from scratch; worse, with M particles stepping concurrently from the
    empty context every particle finds ``_initial_beam is None`` and builds its own. Profiled on
    BBH cross-model: 20 trie builds per example, 77 % of wall time, GPU idle. The trie depends
    only on the vocabulary and the EOS strings, so this caches it per model (the byte-level
    results are unchanged: the trie is deterministic). Library fix: build it in ``ByteLLM.__init__``.
    """
    from genlm.bytes import AsyncTokenByteTrie, ByteBeamState
    from genlm.bytes.byte_lm.trie_state import LazyTrieState, TrieMode

    if getattr(ByteBeamState, "_trie_cache", None) is not None:
        return
    ByteBeamState._trie_cache = {}

    async def initial(cls, llm, params, trie_opts=None):
        key = (id(llm), tuple(sorted(params.eos_byte_strings)), tuple(sorted((trie_opts or {}).items())))
        trie = cls._trie_cache.get(key)
        if trie is None:
            trie = cls._trie_cache[key] = AsyncTokenByteTrie.from_vocab(
                llm.byte_vocab, **(trie_opts or {}), eos_byte_strings=params.eos_byte_strings)
        state = LazyTrieState.initial(llm, trie, mode=TrieMode.WITH_EOS)
        return cls([await state.materialize()], params)

    ByteBeamState.initial = classmethod(initial)


BEAM_CACHE = 64   # beams kept per ByteLLM (library default 1024 ~ 100 GB of host memory per model at K = 20)


def load_byte_llms(model_1, model_2, beam_k=BEAM_K, prune_threshold=PRUNE_THRESHOLD, batch_size=20, cache_size=BEAM_CACHE):
    """Two ``ByteLLM``s for byte-level ensembling, one model per visible GPU, HF backend.

    Returns ``(byte_llm_1, byte_llm_2, raw_1, raw_2)``; ``raw_i.tokenizer.bos_token`` is the marker
    to strip from that model's chat-formatted prompt (see ``strip_bos``).

    ``batch_size`` is the HF backend's auto-batching cap (queries per forward pass). ``cache_size``
    is ByteLLM's LRU of beams: a miss rebuilds the beam from the longest cached prefix, so a small
    cache (>= a few times the particle count) changes nothing but memory -- each cached beam holds
    K materialised next-byte mass vectors (~100 MB at K = 20).

    Byte level needs the *full* next-token distribution: any top-k truncation (even 65536)
    starves the byte trie and prefill returns an empty beam. The HF backend returns it
    directly. Two 8B models in bf16 fit a pair of 24 GB cards.
    """
    from genlm.backend.llm import AsyncTransformer

    cache_byte_tries()
    g1, g2 = _layout(model_1, model_2)
    raw1 = AsyncTransformer.from_name(model_1, hf_opts=_hf_opts(g1), batch_size=batch_size)
    raw2 = raw1 if model_2 == model_1 else AsyncTransformer.from_name(model_2, hf_opts=_hf_opts(g2), batch_size=batch_size)
    for raw in (raw1, raw2):
        clamp_logprobs_to_vocab(raw)
        # The byte beam starts from the tokenizer's BOS; Qwen has none, so the paper's runners
        # set it on the model's own tokenizer. Returned so the caller strips the same marker.
        qwen_bos(raw.tokenizer)

    def wrap(raw):
        return ByteLLM(raw, BeamParams(K=beam_k, prune_threshold=prune_threshold, eos_byte_strings=[_eos_bytes(raw)]),
                       cache_size=cache_size)

    return wrap(raw1), wrap(raw2), raw1, raw2


def clear_backend_caches(*raws) -> None:
    """Drop the HF backend's per-prefix memo (next-token log-probs and KV of every visited prefix).

    FLAG (library): ``AsyncTransformer`` memoises every token prefix it has scored in a trie that is
    never pruned; byte-level SMC visits thousands of prefixes per example (one row per beam
    state per byte), so host memory grows by gigabytes per example and a 100-example run reached
    120 GB (measured: +1.1 GB per BBH example without clearing, flat at 6-7 GB with it). Prompts
    differ between examples, so nothing is lost by clearing; the next example re-scores its prompt
    (20-30 % of its time), hence the runner clears every few examples.
    """
    for raw in dict.fromkeys(raws):
        if hasattr(raw, "clear_cache"):
            raw.clear_cache()


def release_gpu_after_error(exc: BaseException) -> None:
    """After a caught CUDA OOM, drop the traceback and return the memory.

    The exception keeps every frame -- and so every beam tensor -- reachable; without
    this the next example OOMs sooner and a cell wedges after a few failures.
    """
    import gc

    exc.__traceback__ = None
    del exc
    gc.collect()
    if torch.cuda.is_available():
        for d in range(torch.cuda.device_count()):
            with torch.cuda.device(d):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        logger.info(
            "post-error cleanup: "
            + ", ".join(
                f"cuda:{d} reserved={torch.cuda.memory_reserved(d) / 2**30:.2f}GiB"
                for d in range(torch.cuda.device_count())
            )
        )
