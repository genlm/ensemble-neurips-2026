"""One module per dataset. Each exposes a ``Task`` class with the same surface:

    task = Task(n_query, base_seed, **task_kwargs)
    task.items                              # the evaluation instances, fixed order
    task.prompt(item, template, tokenizer, slot)   # chat-formatted prompt string
    task.score(item, text) -> (is_correct: bool, info)
    task.record(item) -> dict               # identifying fields for the output line
    Task.templates                          # the paper's (template_1, template_2)
    Task.max_tokens                         # {"token": n, "byte": n}
    Task.max_model_len
    Task.temperature

``slot`` is 1 or 2 (which model the prompt is for); only mmlu uses it, because its
few-shot draw depends on the slot and not on the template name.
"""

import importlib

NAMES = ("bbh", "spider", "json_schema", "gsm8k", "mmlu")


def get_task_class(name):
    if name not in NAMES:
        raise ValueError(f"unknown task {name!r}; choose from {NAMES}")
    return importlib.import_module(f"ensemble_bench.tasks.{name}").Task
