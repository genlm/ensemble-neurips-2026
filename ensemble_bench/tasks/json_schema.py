"""JSON Schema (JSONSchemaBench / Glaiveai2K) via genlm-eval. Prompt styles are the paper's."""

import json

import numpy as np
from datasets import load_dataset
from genlm.eval.domains.json_schema import (
    DEFAULT_SYSTEM_PROMPT,
    JSONSchemaBenchDataset,
    JSONSchemaBenchEvaluator,
    default_prompt_formatter,
)

SYSTEM_PROMPTS = {
    "default": DEFAULT_SYSTEM_PROMPT,
    "detailed": (
        "You are an expert at generating valid JSON objects. Given a JSON schema, "
        "your task is to generate a JSON object that perfectly matches all constraints "
        "specified in the schema including types, required fields, patterns, enums, and "
        "validation rules. Output the JSON object on a single line with no additional text."
    ),
    "concise": "Generate valid JSON matching the schema. Single line output only.",
    "minimal": "Generate JSON from schema:",
    "technical": (
        "Task: JSON object generation from schema specification.\n"
        "Input: JSON Schema (draft-04 or later)\n"
        "Output: Valid JSON object satisfying all schema constraints\n"
        "Format: Single-line JSON, no extraneous text"
    ),
    "persona": (
        "You are a JSON generation API endpoint. You receive a JSON schema and "
        "respond with exactly one valid JSON object that conforms to it, emitted on "
        "a single line. You never return prose, explanations, or markdown - only the "
        "raw JSON object, as an API would."
    ),
}


class Task:
    name = "json_schema"
    templates = ("default", "technical")
    max_tokens = {"token": 300, "byte": 512}
    max_model_len = 2048
    temperature = 1.0

    def __init__(self, n_query=100, base_seed=42, subtasks=("Glaiveai2K",), split="val"):
        schemas, tasks, ids = [], [], []
        for t in subtasks:
            ds = load_dataset("epfl-dlab/JSONSchemaBench", t)[split]
            for schema, uid in zip(ds["json_schema"], ds["unique_id"]):
                schemas.append(json.loads(schema))
                tasks.append(t)
                ids.append(f"{t}_{uid}")
        full = list(JSONSchemaBenchDataset(schemas, tasks, ids))
        if n_query is not None and n_query < len(full):
            idx = sorted(np.random.RandomState(base_seed).choice(len(full), size=n_query, replace=False))
            full = [full[i] for i in idx]
        self.items = full
        self.evaluator = JSONSchemaBenchEvaluator()

    def prompt(self, item, template, tokenizer, slot=1):
        ids = default_prompt_formatter(tokenizer, item, True, SYSTEM_PROMPTS[template])
        return tokenizer.decode(ids)

    def score(self, item, text):
        r = self.evaluator.evaluate_sample(item, text)
        return float(r.score) >= 1.0, r.desc

    def record(self, item):
        return {"subtask": item.task, "instance_id": str(item.instance_id), "schema": item.json_schema}
