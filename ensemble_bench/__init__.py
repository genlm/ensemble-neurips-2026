"""Replication harness for "Ensembling Language Models with Sequential Monte Carlo".

The library (``genlm.control.Ensemble``, ``ByteLLM``, ``direct_token_sampler``) is a
dependency; this package only holds what the experiments need on top of it:
dataset loading, the paper's prompt templates, scoring, and one SMC runner.
"""
