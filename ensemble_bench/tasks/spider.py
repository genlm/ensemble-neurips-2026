"""Spider text-to-SQL, execution accuracy. Prompt templates are the paper's ``bad_1`` / ``mid_1``.

Needs the Spider release unpacked at ``$SPIDER_DIR`` (``tables.json``, ``dev.json``,
``train_spider.json``, ``database/``); it is CC BY-SA and not redistributed here.
"""

import os
from pathlib import Path

import numpy as np

from ensemble_bench.spider.dialogue import load_spider_data
from ensemble_bench.spider.evaluator import Evaluator
from ensemble_bench.spider.prompt_formatter import SpiderPromptFormatterExtended, format_prompt
from ensemble_bench.spider.schema import load_schemas

# name -> (content_format, schema_format, normalization, limit_value, few_shot, in_domain,
#          n_shot, single_schema, complement_flag)
TEMPLATES = {
    "bad_1": (None, "Table(Columns)", True, 0, False, False, 0, True, 0),       # table + column names only
    "mid_1": (None, "Columns=[]+FK", True, 0, True, True, 1, True, 0),          # + foreign keys, 1 in-domain shot
    "mid_1_bytesafe": (None, "Columns=[]+FK", True, 0, False, False, 0, True, 0),
    # K-expert ablation (the paper's nexpert runner); its mid_1 was 3-shot "Table(Columns)+FK"
    "bad_2": (None, "Columns=[]", True, 0, False, False, 0, True, 0),             # column lists only
    "mid_2": (None, "Table(Columns)+FK", True, 0, True, False, 3, None, 0),       # 3 out-of-domain shots
    "good_1": ("SelectCol", "Table(Columns)+FK", True, 3, True, True, 3, True, 0),  # + column value examples
    "good_2": ("SelectRow", "Table(Columns)+FK", True, 3, True, True, 3, True, 0),  # + row examples
}


class Task:
    name = "spider"
    templates = ("bad_1", "mid_1")
    max_tokens = {"token": 1024, "byte": 1024}
    max_model_len = 2048
    temperature = 1.0

    def __init__(self, n_query=100, base_seed=42, spider_dir=None):
        self.dir = Path(spider_dir or os.environ.get("SPIDER_DIR", "data/spider"))
        self.schemas = load_schemas(schemas_path=self.dir / "tables.json", db_path=self.dir / "database")
        self.dev_full = load_spider_data(self.dir / "dev.json")
        self.train = load_spider_data(self.dir / "train_spider.json")
        if n_query is None or n_query >= len(self.dev_full):
            self.items = self.dev_full
        else:
            # the paper's runner subsampled with a fixed RandomState(42), independent of base_seed
            idx = sorted(np.random.RandomState(42).choice(len(self.dev_full), size=n_query, replace=False))
            self.items = [self.dev_full[i] for i in idx]
        self.evaluator = Evaluator(self.dir)
        self._formatters = {}

    def _formatter(self, template):
        if template not in self._formatters:
            content, schema, norm, limit, *_ = TEMPLATES[template]
            self._formatters[template] = SpiderPromptFormatterExtended(
                spider_train_data=self.train, db_map=self.schemas, spider_dev_data=self.dev_full,
                data_dir=str(self.dir / "database"), db_content_format=content, schema_format=schema,
                normalization=norm, limit_value=limit,
            )
        return self._formatters[template]

    def prompt(self, item, template, tokenizer, slot=1):
        _, _, _, _, few_shot, in_domain, n_shot, single_schema, complement = TEMPLATES[template]
        return format_prompt(item, self._formatter(template), tokenizer, few_shot=few_shot, in_domain=in_domain,
                             n_shot=n_shot, single_schema=single_schema, complement_flag=complement)

    def score(self, item, text):
        ok, reason = self.evaluator.evaluate(item.query, text, item.schema_name)
        return bool(ok), reason

    def record(self, item):
        return {"question": item.utterance, "gold": item.query, "db_name": item.schema_name}
