"""Execution-accuracy wrapper around the official Spider evaluator.

The Yale code (``third_party/spider``, a git submodule pinned at ``b7b5b8c``) is used
unmodified. This wrapper adds what the paper's runs needed on top of it: stripping
markdown fences and trailing semicolons from model output, running the SQL in a
subprocess with an optional timeout (generated queries can hang sqlite), and reporting
*why* a prediction failed (``invalid`` = unparsable, ``mismatch`` = wrong result).
"""

import multiprocessing
import sys
from pathlib import Path

import nltk

_YALE = Path(__file__).resolve().parents[2] / "third_party" / "spider"
if not (_YALE / "evaluation.py").exists():
    raise ImportError(
        f"Spider evaluator not found at {_YALE}; run `git submodule update --init third_party/spider`"
    )
if str(_YALE) not in sys.path:
    sys.path.insert(0, str(_YALE))

import evaluation as E  # noqa: E402  (Yale, imports its sibling process_sql)
from evaluation import (  # noqa: E402
    build_foreign_key_map_from_json,
    build_valid_col_units,
    eval_exec_match,
    rebuild_sql_col,
    rebuild_sql_val,
)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


def _exec_match_worker(db, p_str, g_str, pred, gold, q):
    try:
        q.put(eval_exec_match(db, p_str, g_str, pred, gold))
    except Exception:
        q.put(False)


def exec_match_with_timeout(db, p_str, g_str, pred, gold, timeout=None):
    q = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_exec_match_worker, args=(db, p_str, g_str, pred, gold, q))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return False
    return q.get() if not q.empty() else False


class Evaluator:
    def __init__(self, spider_dir, timeout=None):
        spider_dir = Path(spider_dir)
        self.db_path = spider_dir / "database"
        self.kmaps = build_foreign_key_map_from_json(spider_dir / "tables.json")
        self.official = E.Evaluator()
        self.timeout = timeout

    def evaluate(self, gold, pred, db_name, return_level=False):
        """``(True, None)`` on execution match, else ``(False, 'invalid' | 'mismatch')``."""
        db = self.db_path / db_name / f"{db_name}.sqlite"
        schema = E.Schema(E.get_schema(str(db)))
        pred = pred.replace("```sql", "").replace("```", "").strip().rstrip(";") + ";"
        try:
            g_sql = E.get_sql(schema, gold)
            p_sql = E.get_sql(schema, pred)
        except Exception:
            return False, "invalid"

        kmap = self.kmaps[db_name]
        g_sql = rebuild_sql_col(build_valid_col_units(g_sql["from"]["table_units"], schema), rebuild_sql_val(g_sql), kmap)
        p_sql = rebuild_sql_col(build_valid_col_units(p_sql["from"]["table_units"], schema), rebuild_sql_val(p_sql), kmap)

        match = exec_match_with_timeout(str(db), pred, gold, p_sql, g_sql, timeout=self.timeout)
        reason = None if match else "mismatch"
        if not return_level:
            return match, reason
        return match, reason, self.official.eval_hardness(g_sql)
