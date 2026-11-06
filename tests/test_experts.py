"""Nested binary ensembles equal the uniform K-ary operator (the paper's ``EnsembleN``)."""

import asyncio

import numpy as np
from genlm.control.potential.base import Potential

from ensemble_bench.experts import k_expert_ensemble, expert_logws

VOCAB = [b"a", b"b", b"c"]


class Toy(Potential):
    """A fixed random bigram model over VOCAB (+ EOS), proper so prefix/complete are consistent."""

    def __init__(self, seed):
        super().__init__(vocabulary=VOCAB)
        rng = np.random.default_rng(seed)
        self.rows = {}
        self.rng = rng

    def row(self, context):
        key = tuple(context)
        if key not in self.rows:
            p = self.rng.dirichlet(np.ones(len(self.vocab_eos)))
            self.rows[key] = np.log(p)
        return self.rows[key]

    async def prefix(self, context):
        return float(sum(self.row(context[:i])[self.lookup[t]] for i, t in enumerate(context)))

    async def complete(self, context):
        return float(await self.prefix(context) + self.row(context)[self.lookup[self.eos]])

    async def logw_next(self, context):
        return self.make_lazy_weights(self.row(context))


def nary(op, ws):
    ws = np.asarray(ws)
    return ws.min(axis=0) if op == "min" else ws.mean(axis=0)


def test_nested_equals_nary(op="min", K=3):
    pots = [Toy(seed) for seed in range(K)]
    ens = k_expert_ensemble(pots, op)
    rng = np.random.default_rng(0)
    for _ in range(20):
        ctx = [VOCAB[i] for i in rng.integers(0, len(VOCAB), size=rng.integers(0, 5))]
        want_prefix = nary(op, [asyncio.run(p.prefix(ctx)) for p in pots])
        want_complete = nary(op, [asyncio.run(p.complete(ctx)) for p in pots])
        assert np.isclose(asyncio.run(ens.prefix(ctx)), want_prefix)
        assert np.isclose(asyncio.run(ens.complete(ctx)), want_complete)
        # next-token weights: prefix(ctx + t) - prefix(ctx) of the K-ary target
        got = asyncio.run(ens.logw_next(ctx)).weights
        want = np.array([nary(op, [asyncio.run(p.prefix(ctx + [t])) if t is not ens.eos
                                   else asyncio.run(p.complete(ctx)) for p in pots]) for t in ens.vocab_eos]) - want_prefix
        assert np.allclose(got, want)
        assert np.allclose(asyncio.run(expert_logws(ens, ctx)), [asyncio.run(p.prefix(ctx)) for p in pots])


if __name__ == "__main__":  # no pytest needed
    for op in ("min", "prod"):
        for K in (2, 3, 4, 6):
            test_nested_equals_nary(op, K)
            print(f"ok: nested {op} == {K}-ary {op}")
