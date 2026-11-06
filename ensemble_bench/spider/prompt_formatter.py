"""Spider prompt construction used in the paper (templates ``bad_1`` and ``mid_1``).

``SpiderPromptFormatterExtended`` serialises a database schema (table/column names, optionally
foreign keys) and builds zero-shot or in-domain few-shot chat prompts; ``format_prompt`` applies
the model's chat template. The schema helpers come from Microsoft's CoSQL/Spider code (MIT).
"""

import os
import random
import sqlite3
from collections import defaultdict
from typing import Dict, List, Optional

import sqlparse

from ensemble_bench.spider.dialogue import SpiderDatum
from ensemble_bench.spider.schema import DbSchema


def get_sql_query(example: SpiderDatum, normalize: bool = False) -> str:
    if hasattr(example, "gold"):
        query = example.gold
    elif hasattr(example, "query"):
        query = example.query
    else:
        raise AttributeError("SpiderDatum object has no 'gold' or 'query' attribute.")
    return normalize_sql(query) if normalize else query + ";"


def normalize_sql(query: str) -> str:
    return sqlparse.format(query, keyword_case="lower", identifier_case="lower")


def format_prompt(
    datum,
    formatter,
    tokenizer,
    few_shot: bool = True,
    in_domain: bool = False,
    n_shot: int = 3,
    single_schema: bool = False,
    complement_flag: Optional[int] = None,
) -> str:
    """Build the chat messages for ``datum`` and render them with the tokenizer's chat template."""
    model_name = getattr(tokenizer, "name_or_path", "") or ""
    is_gemma = "gemma" in model_name.lower()

    def convert_system_to_user(messages):
        """Gemma has no system role: fold system content into the first user turn."""
        if not is_gemma:
            return messages
        converted, system_parts = [], []
        for msg in messages:
            if msg.get("role") == "system":
                system_parts.append(msg["content"])
            elif system_parts and msg.get("role") == "user":
                converted.append({"role": "user", "content": "\n\n".join(system_parts) + "\n\n" + msg["content"]})
                system_parts = []
            else:
                converted.append(msg)
        if system_parts:
            converted.insert(0, {"role": "user", "content": "\n\n".join(system_parts)})
        return converted

    if few_shot and in_domain:
        messages = formatter.format_single_domain_few_shot_prompt(
            datum, n_shot=n_shot, single_schema=single_schema, complement_flag=complement_flag
        )
    elif few_shot:
        messages = formatter.format_cross_domain_few_shot_prompt(datum, n_shot=n_shot, complement_flag=complement_flag)
    else:
        messages = formatter.format_zero_shot_prompt(datum, complement_flag=complement_flag)
    messages = convert_system_to_user(messages)
    return tokenizer.decode(tokenizer.apply_chat_template(messages, add_generation_prompt=True))


def get_foreign_keys(db_id: str, table: str, create_table_statement: str) -> List[str]:
    """
    Parses CREATE TABLE SQL to extract foreign key constraints.

    Args:
        db_id: Database identifier (not used, kept for consistency).
        table: Table name.
        create_table_statement: Raw SQL for the CREATE TABLE statement.

    Returns:
        A list of foreign key mappings in the format "table.col = ref_table.ref_col".
    """
    foreign_keys = []
    for row in create_table_statement.split("\n"):
        if (
            row.lower().startswith("foreign key")
            and row.lower().count("foreign key") == 1
        ):
            row = row.split(" on ")[0].split(" ON ")[0]
            row = row.replace(",", " ").replace("(", "  ").replace(")", "  ")
            row = row.split()
            if len(row) != 6:
                for i, tok in enumerate(row):
                    if tok.lower() == "references":
                        references_pos = i
                num_keys = references_pos - 2
                for i in range(num_keys):
                    key = f"{table}.{row[2 + i]} = {row[4 + num_keys - 1]}.{row[4 + num_keys + i]}"
                    foreign_keys.append(key)
            else:
                key = f"{table}.{row[2]} = {row[4]}.{row[5]}"
                foreign_keys.append(key)
    return foreign_keys


def extract_tablecolumn_prompt(
    prompt_db: str,
    db_id: str,
    db_path: str,
    limit_value: int = 3,
    normalization: bool = True,
    table_subset: Optional[List[str]] = None
) -> str:
    """
    Extracts table and column information (optionally foreign keys) in a simplified prompt format.

    Args:
        prompt_db: Format of the prompt ("Table(Columns)", "Columns=[]", etc.).
        db_id: Database identifier.
        db_path: Path to the SQLite DB.
        limit_value: Row sample size for inference.
        normalization: Whether to lowercase identifiers.

    Returns:
        A formatted string with table and column info and optional foreign key definitions.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    tables = cursor.execute(
        "SELECT * FROM sqlite_master WHERE type='table';"
    ).fetchall()

    prompt = ""
    foreign_keys = []

    for table in tables:
        if table[1] not in table_subset:
            continue
        table_name = table[1]
        create_table_statement = table[-1]
        if normalization:
            table_name = table_name.lower()

        create_table_statement = normalize_create_table(
            table_name, create_table_statement
        )
        foreign_keys_one_table = get_foreign_keys(
            db_id, table_name, create_table_statement
        )

        table_info_query = f"PRAGMA table_info({table_name});"
        headers = [x[1] for x in cursor.execute(table_info_query).fetchall()]
        if normalization:
            headers = [h.lower() for h in headers]
            foreign_keys_one_table = [fk.lower() for fk in foreign_keys_one_table]

        foreign_keys.extend(foreign_keys_one_table)

        if prompt_db.startswith("Table(Columns)"):
            prompt += f"{table_name}({', '.join(headers)});\n"
        elif prompt_db.startswith("Columns=[]"):
            prompt += f"Table {table_name}, Columns = [{', '.join(headers)}];\n"
        elif prompt_db.startswith("+FK"):
            prompt += ""

    if "+FK" in prompt_db:
        prompt += "Foreign_keys = [" + ", ".join(foreign_keys) + "];\n"

    prompt += "\n"
    conn.close()
    return prompt


class SpiderPromptFormatterExtended:
    """
    Formats prompts for text-to-SQL models using the Spider dataset with different levels of in-context information.
    """

    def __init__(
        self,
        spider_train_data: List[SpiderDatum],
        db_map: Dict[str, DbSchema],
        spider_dev_data: List[SpiderDatum] = None,
        data_dir: Optional[str] = None,
        db_content_format: Optional[str] = None,
        schema_format: str = "CreateTable",
        normalization: bool = True,
        limit_value: int = 3,
        wrap_sql_in_code_block: bool = False,
    ):
        if db_content_format is not None:
            raise ValueError("database-content formats are not supported (the paper's templates use none)")
        if schema_format not in (
            "CreateTable",
            "Table(Columns)",
            "Table(Columns)+FK",
            "Columns=[]",
            "Columns=[]+FK",
            "+FK",
            None,
        ):
            raise ValueError(f"Invalid schema_format: {schema_format}")
        self.spider_dev_data = (
            spider_dev_data  # may be needed for in-domain few-shot examples
        )
        self.spider_train_data = spider_train_data
        self.db_map = db_map
        self.data_dir = data_dir
        self.db_content_format = db_content_format
        self.schema_format = schema_format
        self.normalization = normalization
        self.limit_value = limit_value
        self.wrap_sql_in_code_block = wrap_sql_in_code_block

        self.system_prompt = (
            "You are a coding assistant helping an analyst answer questions over business data in SQL. "
            "More specifically, the analyst provides you a database schema (tables in the database along with their column names and types) and asks a question about the data that can be solved by issuing a SQL query to the database. "
            "In response, you write the SQL statement that answers the question. You do not provide any commentary or explanation of what the code does, just the SQL statement ending in a semicolon."
        )

    def _generate_db_prompt(self, schema_name: str, complement_flag: Optional[int] = None) -> str:
        db_path = os.path.join(self.data_dir, schema_name, schema_name + ".sqlite")

        def filter_tables_by_half(tables):
            mid = len(tables) // 2
            if complement_flag == 1:
                return tables[:mid]
            elif complement_flag == 2:
                return tables[mid:]
            return tables

        # Custom versions of schema extractors with table filtering
        def extract_filtered_schema():
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            all_tables = [row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
            filtered_tables = filter_tables_by_half(all_tables)
            conn.close()
            return filtered_tables

        filtered_tables = extract_filtered_schema()

        if self.schema_format is None:
            return ""
        if not self.schema_format.startswith(("Table(Columns)", "Columns=[]", "+FK")):
            # the paper's templates only use these schema formats; the CreateTable and
            # database-content variants of the original code are not carried over
            raise ValueError(f"Unsupported schema format: {self.schema_format}")
        return extract_tablecolumn_prompt(
            self.schema_format,
            schema_name,
            db_path,
            limit_value=self.limit_value,
            normalization=self.normalization,
            table_subset=filtered_tables,
        )


    def format_zero_shot_prompt(self, datum: SpiderDatum, complement_flag:Optional[int] = None) -> List[Dict[str, str]]:
        """
        Formats a zero-shot prompt (no examples, just schema and question).

        Args:
            datum: SpiderDatum instance containing the question and schema reference.

        Returns:
            List of formatted messages for the model.
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        schema_prompt = self._generate_db_prompt(datum.schema_name, complement_flag=complement_flag)
        if self.schema_format is None:
            user_message = (
                f"""Please write me a SQL statement that answers the following question: {datum.utterance}"""
                """\n\nRemember, DO NOT provide any commentary or explanation of what the code does, just the SQL statement ending in a semicolon."""
            )
        elif self.schema_format.startswith("+FK"):
            user_message = (
                f"""Here are the foreign keys of the database schema:\n\n{schema_prompt} Please write me a SQL statement that answers the following question: {datum.utterance}"""
                """\n\nRemember, DO NOT provide any commentary or explanation of what the code does, just the SQL statement ending in a semicolon."""
            )
        else:
            user_message = (
                f"""Here is a database schema:\n\n{schema_prompt} Please write me a SQL statement that answers the following question: {datum.utterance}"""
                """\n\nRemember, DO NOT provide any commentary or explanation of what the code does, just the SQL statement ending in a semicolon."""
            )
        messages.append({"role": "user", "content": user_message})
        return messages

    def format_single_domain_few_shot_prompt(
        self, datum: SpiderDatum, n_shot: int = 3, single_schema: bool = False, complement_flag:Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Formats a few-shot prompt where all examples come from the same database schema.

        Args:
            datum: The final test example.
            in_context_examples_ids: List of indices pointing to demonstration examples.

        Returns:
            List of formatted messages with shared schema and examples.
        """
        schema_prompt = self._generate_db_prompt(datum.schema_name, complement_flag=complement_flag)
        same_schema_examples = [
            (i, ex)
            for i, ex in enumerate(self.spider_dev_data)
            if ex.schema_name == datum.schema_name and ex != datum
        ]
        # Deterministically select the shortest examples (by query length)
        if len(same_schema_examples) == 0:
            selected = []
        elif len(same_schema_examples) <= n_shot:
            selected = same_schema_examples
        else:
            # Sort by query length (shortest first), then take first n_shot
            # Use stable sort by index as tiebreaker for determinism
            selected = sorted(
                same_schema_examples,
                key=lambda x: (len(get_sql_query(x[1])), x[0])
            )[:n_shot]

        in_context_examples_ids = [i for i, _ in selected]

        for ct, idx in enumerate(in_context_examples_ids):
            example = self.spider_dev_data[idx]
            if self.schema_format is None:
                user_message = (
                    f"""Please write me a SQL statement that answers the following question: {example.utterance}"""
                    """\n\nRemember, DO NOT provide any commentary or explanation of what the code does, just the SQL statement ending in a semicolon."""
                )
            elif self.schema_format.startswith("+FK"):
                if ct == 0:
                    system_prompt_with_schema = self.system_prompt + f"\n Here is the database schema for all following queries: \n\n{schema_prompt}"
                if not single_schema:
                    user_message = (
                        f"""Here are the foreign keys of the database schema:\n\n{schema_prompt} Please write me a SQL statement that answers the following question: {example.utterance}"""
                        """\n\nRemember, DO NOT provide any commentary or explanation of what the code does, just the SQL statement ending in a semicolon."""
                    )
                else:
                    user_message = (
                        f"""Please write me a SQL statement that answers the following question: {example.utterance}"""
                        """\n\nRemember, DO NOT provide any commentary or explanation of what the code does, just the SQL statement ending in a semicolon."""
                    )
            else:
                if ct == 0:
                    system_prompt_with_schema = self.system_prompt + f"\n Here is the database schema for all following queries: \n\n{schema_prompt}"
                if not single_schema:
                    user_message = (
                        f"""Here is a database schema:\n\n{schema_prompt}"""
                        f"""Please write me a SQL statement that answers the following question: {example.utterance}"""
                        """\n\nRemember, DO NOT provide any commentary or explanation of what the code does, just the SQL statement ending in a semicolon."""
                    )
                else:
                    user_message = (
                        f"""Please write me a SQL statement that answers the following question: {example.utterance}"""
                        """\n\nRemember, DO NOT provide any commentary or explanation of what the code does, just the SQL statement ending in a semicolon."""
                    )
            if ct == 0:
                if single_schema:
                    messages = [{"role": "system", "content": system_prompt_with_schema}]
                else:
                    messages = [{"role": "system", "content": self.system_prompt}]
            messages.append({"role": "user", "content": user_message})
            sql_query = get_sql_query(example, normalize=self.normalization) + ";"
            if self.wrap_sql_in_code_block:
                sql_query = f"```sql\n{sql_query}\n```"
            messages.append(
                {
                    "role": "assistant",
                    "content": sql_query,
                }
            )

        # Final prompt
        if self.schema_format is None:
            user_message = (
                f"""Please write me a SQL statement that answers the following question: {datum.utterance}"""
                """\n\nRemember, DO NOT provide any commentary or explanation of what the code does, just the SQL statement ending in a semicolon."""
            )
        elif self.schema_format.startswith("+FK"):
            if single_schema:
                user_message = (
                    f"""Please write me a SQL statement that answers the following question: {datum.utterance}"""
                    """\n\nRemember, DO NOT provide any commentary or explanation of what the code does, just the SQL statement ending in a semicolon."""
                )
            else:
                user_message = (
                    f"""Here are the foreign keys of the database schema:\n\n{schema_prompt}"""
                    f"""Please write me a SQL statement that answers the following question: {datum.utterance}"""
                    """\n\nRemember, DO NOT provide any commentary or explanation of what the code does, just the SQL statement ending in a semicolon."""
                )
        else:
            if single_schema:
                user_message = (
                    f"""Please write me a SQL statement that answers the following question: {datum.utterance}"""
                    """\n\nRemember, DO NOT provide any commentary or explanation of what the code does, just the SQL statement ending in a semicolon."""
                )
            else:
                user_message = (
                    f"""Here is a database schema:\n\n{schema_prompt}"""
                    f"""Please write me a SQL statement that answers the following question: {datum.utterance}"""
                    """\n\nRemember, DO NOT provide any commentary or explanation of what the code does, just the SQL statement ending in a semicolon."""
                )
        messages.append({"role": "user", "content": user_message})

        return messages

    def format_cross_domain_few_shot_prompt(
        self, datum: SpiderDatum, n_shot: int = 3, complement_flag:Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Formats a few-shot prompt using demonstrations from different domains.

        Args:
            datum: The test example.
            n_shot: Number of cross-domain examples to include.

        Returns:
            List of messages suitable for prompting a model.
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        db_to_examples = defaultdict(list)
        for example in self.spider_train_data:
            db_to_examples[example.schema_name].append(example)

        test_schema_name = datum.schema_name
        candidate_dbs = [db for db in db_to_examples if db != test_schema_name]
        random.shuffle(candidate_dbs)

        added_shots = 0

        for example_id in [10, 100, 1000][:n_shot]:
            example = self.spider_train_data[example_id]
            schema_prompt = self._generate_db_prompt(example.schema_name, complement_flag=complement_flag)
            if self.schema_format is None:
                user_message = (
                    f"""Please write me a SQL statement that answers the following question: {example.utterance}"""
                    """\n\nRemember, DO NOT provide any commentary or explanation of what the code does, just the SQL statement ending in a semicolon."""
                )
            elif self.schema_format.startswith("+FK"):
                user_message = (
                    f"""Here are the foreign keys of the database schema:\n\n{schema_prompt}"""
                    f"""Please write me a SQL statement that answers the following question: {example.utterance}"""
                    """\n\nRemember, DO NOT provide any commentary or explanation of what the code does, just the SQL statement ending in a semicolon."""
                )
            else:
                user_message = (
                    f"""Here is a database schema:\n\n{schema_prompt}"""
                    f"""Please write me a SQL statement that answers the following question: {example.utterance}"""
                    """\n\nRemember, DO NOT provide any commentary or explanation of what the code does, just the SQL statement ending in a semicolon."""
                )
            messages.append({"role": "user", "content": user_message})
            sql_query = get_sql_query(example, normalize=self.normalization) + ";"
            if self.wrap_sql_in_code_block:
                sql_query = f"```sql\n{sql_query}\n```"
            messages.append(
                {
                    "role": "assistant",
                    "content": sql_query,
                }
            )
            added_shots += 1

        schema_prompt = self._generate_db_prompt(test_schema_name, complement_flag=complement_flag)

        if self.schema_format is None:
            user_message = f"""Please write me a SQL statement that answers the following question: {datum.utterance}"""
        elif self.schema_format.startswith("+FK"):
            user_message = (
                f"""Here are the foreign keys of the database schema:\n\n{schema_prompt}"""
                f"""Please write me a SQL statement that answers the following question: {datum.utterance}"""
                """\n\nRemember, DO NOT provide any commentary or explanation of what the code does, just the SQL statement ending in a semicolon."""
            )
        else:
            user_message = (
                f"""Here is a database schema:\n\n{schema_prompt}"""
                f"""Please write me a SQL statement that answers the following question: {datum.utterance}"""
                """\n\nRemember, DO NOT provide any commentary or explanation of what the code does, just the SQL statement ending in a semicolon."""
            )
        messages.append({"role": "user", "content": user_message})

        return messages


def normalize_create_table(table_name: str, create_table_statement: str) -> str:
    """
    Normalizes a CREATE TABLE statement by lowercasing and fixing table name references.

    Args:
        table_name: Table name to normalize.
        create_table_statement: Raw SQL string of the CREATE TABLE.

    Returns:
        Normalized CREATE TABLE SQL.
    """
    create_table_statement = create_table_statement.lower()
    create_table_statement = create_table_statement.replace(
        table_name.lower(), table_name
    )
    return create_table_statement
