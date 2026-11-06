"""Token-level ensembling baselines: UniTE (Yao et al., 2024) and GaC (Yu et al., 2024).

Ports of the paper's ``genlm/ensemble/base.py`` onto ``genlm.control.Potential``. Both combine the
two models' *next-token* distributions and renormalise, so sampling from ``logw_next`` with
``direct_token_sampler`` is ancestral sampling from the baseline (every particle weight is 0).
They work across tokenizers: the vocabulary is the union of the two models' byte strings (special
tokens removed), and a context token missing from one model's vocabulary is fed to it as single bytes.
"""

import asyncio

import numpy as np
from arsenal.maths import logsumexp
from genlm.control.potential.base import Potential
from genlm.control.util import to_numpy


def _special_tokens(potential):
    """Byte strings of the tokenizer's special tokens (``<|eot_id|>``, ``<|im_end|>``, ...)."""
    model = potential.model
    return {bytes(model.byte_vocab[tid]) for tid in model.tokenizer.all_special_ids if tid < len(model.byte_vocab)}


def _token_map(potential):
    """bytes -> the model's vocabulary entry. Vocabulary entries are ``Token`` objects (equal by
    token id, so two models' entries never coincide); the union is built on their byte strings,
    and a byte string shared by several tokens of one model maps to its first."""
    out = {}
    for t in potential.vocab:
        out.setdefault(bytes(t), t)
    return out


def _retokenize(context, tokens, eos):
    """A union-vocab context (bytes) in one model's own tokens; unknown byte strings as single bytes."""
    out = []
    for b in context:
        if b is eos:
            out.append(eos)
        elif b in tokens:
            out.append(tokens[b])
        else:
            out.extend(tokens[bytes([x])] for x in b)
    return out


def _gather(weights, idxs):
    """Model log-weights in union-vocab order; tokens the model lacks get -inf."""
    out = np.full(len(idxs), -np.inf)
    for i, idx in enumerate(idxs):
        if idx is not None:
            out[i] = weights[idx]
    return out


def _log_softmax(x):
    return x - logsumexp(x)


class _PairBaseline(Potential):
    """Shared plumbing: union vocabulary, per-model contexts, weighted-mean prefix/complete."""

    def __init__(self, p1, p2, a):
        if not 0 < a < 1:
            raise ValueError("a must be in (0, 1)")
        self.p1, self.p2, self.a = p1, p2, a
        self.tok1, self.tok2 = _token_map(p1), _token_map(p2)
        special = _special_tokens(p1) | _special_tokens(p2)
        super().__init__(vocabulary=[b for b in dict.fromkeys([*self.tok1, *self.tok2]) if b not in special])
        self.p1_idxs = [p1.lookup.get(self.tok1.get(b)) for b in self.vocab] + [p1.lookup[p1.eos]]
        self.p2_idxs = [p2.lookup.get(self.tok2.get(b)) for b in self.vocab] + [p2.lookup[p2.eos]]

    def _ctx(self, context):
        return _retokenize(context, self.tok1, self.p1.eos), _retokenize(context, self.tok2, self.p2.eos)

    async def component_logws(self, context):
        """Each model's log weight of ``context`` (``complete`` if it ends in EOS, else ``prefix``)."""
        c1, c2 = self._ctx(context[:-1] if context and context[-1] is self.eos else context)
        f = (self.p1.complete, self.p2.complete) if context and context[-1] is self.eos else (self.p1.prefix, self.p2.prefix)
        w1, w2 = await asyncio.gather(f[0](c1), f[1](c2))
        return float(w1), float(w2)

    async def prefix(self, context):
        w1, w2 = await self.component_logws(context)
        return logsumexp([np.log(self.a) + w1, np.log(1 - self.a) + w2])

    async def complete(self, context):
        w1, w2 = await self.component_logws([*context, self.eos])
        return logsumexp([np.log(self.a) + w1, np.log(1 - self.a) + w2])

    async def logw_next(self, context):
        batch = await self.batch_logw_next([context])
        return batch.spawn(batch.weights[0])

    async def batch_logw_next(self, contexts):
        pairs = [self._ctx(c) for c in contexts]
        W1, W2 = await asyncio.gather(self.p1.batch_logw_next([c1 for c1, _ in pairs]),
                                      self.p2.batch_logw_next([c2 for _, c2 in pairs]))
        rows1, rows2 = to_numpy(W1.weights), to_numpy(W2.weights)
        return self.make_lazy_weights(np.stack([
            self._merge(_gather(rows1[n], self.p1_idxs), _gather(rows2[n], self.p2_idxs)) for n in range(len(contexts))]))


class UniTE(_PairBaseline):
    """Union of each model's top-k next tokens; per-model softmax over the union; weighted mean."""

    def __init__(self, p1, p2, k=50, a=0.5):
        super().__init__(p1, p2, a)
        self.k = k

    def _merge(self, logw1, logw2):
        k = min(self.k, len(self.vocab_eos))
        top = set(np.argpartition(logw1, -k)[-k:].tolist()) | set(np.argpartition(logw2, -k)[-k:].tolist())
        idx = np.array(sorted(top))
        merged = logsumexp([np.log(self.a) + _log_softmax(logw1[idx]), np.log(1 - self.a) + _log_softmax(logw2[idx])], axis=0)
        out = np.full(len(self.vocab_eos), -np.inf)
        out[idx] = merged
        return out


class GaC(_PairBaseline):
    """Weighted mean of the two softmax-normalised next-token distributions over the union vocabulary.

    ``threshold < 1`` is key-token ensembling: when model 1's max probability exceeds it, model 1's
    distribution is used alone for that step.
    """

    def __init__(self, p1, p2, a=0.5, threshold=1.0):
        super().__init__(p1, p2, a)
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        self.threshold = threshold

    def _merge(self, logw1, logw2):
        logp1, logp2 = _log_softmax(logw1), _log_softmax(logw2)
        if self.threshold < 1 and np.exp(logp1.max()) > self.threshold:
            return logp1
        return logsumexp([np.log(self.a) + logp1, np.log(1 - self.a) + logp2], axis=0)


BASELINES = {
    "unite": lambda p1, p2, a: UniTE(p1, p2, k=50, a=a),
    "gac": lambda p1, p2, a: GaC(p1, p2, a=a, threshold=1.0),
    "gac_key": lambda p1, p2, a: GaC(p1, p2, a=a, threshold=0.9),
}
