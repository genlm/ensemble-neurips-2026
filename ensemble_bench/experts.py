"""K-expert ensembles from nested binary ``Ensemble``s.

The paper's K-expert ablation combines K prompts of one model with a uniform K-ary operator:
``min`` of the K log weights, or their mean (``prod``). Nesting the library's binary ``Ensemble``
reproduces these exactly:

* ``min``: ``Ensemble(Ensemble(p1, p2, "min", 0.5), p3, "min", 0.5)`` is ``min(p1, p2, p3)``
  (the weighted extremum at a = 0.5 is the plain minimum);
* ``prod``: ``Ensemble(E_{k-1}, p_k, "prod", (k-1)/k)`` is ``(k-1)/k * mean_{k-1} + 1/k * p_k``,
  the uniform mean of all k log weights.

``logw_next`` of a nested ensemble is ``prefix(x + [t]) - prefix(x)`` of the K-ary target, so
``direct_token_sampler`` on it is the same SMC as for K = 2.
"""

from genlm.control import Ensemble

NESTABLE = ("min", "prod")


def k_expert_ensemble(potentials, op, cache_size=256):
    """Uniform K-ary ``op`` over ``potentials`` (K >= 2) as nested binary ``Ensemble``s."""
    if op not in NESTABLE:
        raise ValueError(f"K-expert ensembles support {NESTABLE}, not {op!r}")
    ens = potentials[0]
    for k, p in enumerate(potentials[1:], start=2):
        a = 0.5 if op == "min" else (k - 1) / k
        ens = Ensemble(ens, p, op, a, cache_size=cache_size)
    return ens


async def expert_logws(ensemble, context):
    """Every expert's log weight of ``context`` (flattening the nesting), outermost last."""
    if not isinstance(ensemble, Ensemble):
        raise TypeError("expected an Ensemble")
    w1, w2 = await ensemble.component_logws(context)
    if isinstance(ensemble.p1, Ensemble):
        return [*await expert_logws(ensemble.p1, context), w2]
    return [w1, w2]
