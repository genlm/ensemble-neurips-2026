# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from json import load
from typing import Any, Dict, List, Tuple

from ensemble_bench.spider.utils import StrPath


class ColumnType(Enum):
    Text = 'text'
    Number = 'number'
    Time = 'time'
    Boolean = 'boolean'
    Others = 'others'


@dataclass(frozen=True)
class Column:
    name: str
    tpe: ColumnType
    # name in more natural language
    nl_name: str = ''

    @staticmethod
    def star() -> 'Column':
        return Column(name='*', tpe=ColumnType.Text, nl_name='*')


@dataclass(frozen=True)
class ForeignKey:
    column_id: int
    other_column_id: int


@dataclass(frozen=True)
class Table:
    name: str
    columns: List[Column]
    # name in more natural language
    nl_name: str = ''

    def all_columns(self) -> List[Column]:
        return [Column.star()] + self.columns


@dataclass(frozen=True)
class DbSchema:
    name: str
    tables: List[Table]
    # indexes into self.tables
    columns: List[Tuple[int, Column]] = ()  # type: ignore
    # indexes into self.columns
    primary_keys: List[int] = ()  # type: ignore
    # indexes into self.columns
    foreign_keys: List[ForeignKey] = ()  # type: ignore
    # values in the database
    values: List[str] = ()  # type: ignore

    @staticmethod
    def from_json(schema_json: Dict[str, Any], db_path: str) -> 'DbSchema':
        columns: List[Tuple[int, Column]] = [
            (t_id, Column(name=orig, tpe=ColumnType(tpe), nl_name=name))
            for (t_id, orig), (_, name), tpe in zip(
                schema_json['column_names_original'],
                schema_json['column_names'],
                schema_json['column_types'],
            )
        ]
        columns_by_table: Dict[int, List[Column]] = defaultdict(list)
        for t_id, col in columns:
            columns_by_table[t_id].append(col)
        # TODO:
        # tables.json is corrupted in the CoSQL dataset for schema formula_1.
        # "table_names" and "table_names_original" are not collated:
        #     "table_names": [ "races", "drivers", "status", "seasons", "constructors",
        #       "constructor standings", "results", "driver standings", "constructor results", "qualifying",
        #       "circuits", "pitstops", "laptimes" ],
        #     "table_names_original": [ "circuits", "races", "drivers", "status", "seasons", "constructors",
        #       "constructorStandings", "results", "driverStandings", "constructorResults", "qualifying",
        #       "pitStops", "lapTimes" ]
        # in that case, nl_name will be messed up for the table
        tables = [
            Table(name=orig, columns=columns_by_table[t_id], nl_name=name)
            for t_id, (orig, name) in enumerate(
                zip(schema_json['table_names_original'], schema_json['table_names'])
            )
        ]
        foreign_keys = [
            ForeignKey(col, other) for col, other in schema_json['foreign_keys']
        ]
        db_id = schema_json['db_id']
        values: List[str] = []  # column-value picklists are not used by the paper's prompts
        return DbSchema(
            name=schema_json['db_id'],
            columns=columns,
            foreign_keys=foreign_keys,
            primary_keys=schema_json['primary_keys'],
            tables=tables,
            values=values,
        )


def load_schemas(schemas_path: StrPath, db_path: StrPath) -> Dict[str, DbSchema]:
    """Load ``tables.json`` into ``DbSchema`` objects (``db_path`` is the ``database/`` directory)."""
    with open(schemas_path) as tables_file:
        schemas_json = load(tables_file)
    schemas = [DbSchema.from_json(schema_json, str(db_path)) for schema_json in schemas_json]
    return {schema.name: schema for schema in schemas}
