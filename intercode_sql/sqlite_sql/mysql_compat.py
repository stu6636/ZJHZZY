import re
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple


SQLITE_TO_MYSQL_TYPES = {
    "INT": "int",
    "INTEGER": "int",
    "BIGINT": "bigint",
    "SMALLINT": "smallint",
    "TINYINT": "tinyint",
    "VARCHAR": "varchar",
    "CHAR": "char",
    "TEXT": "text",
    "LONGTEXT": "longtext",
    "DATETIME": "datetime",
    "TIMESTAMP": "datetime",
    "DATE": "date",
    "DOUBLE": "double",
    "FLOAT": "float",
    "DECIMAL": "decimal",
    "NUMERIC": "decimal",
    "REAL": "double",
    "BOOLEAN": "tinyint",
    "BOOL": "tinyint",
}


@dataclass(frozen=True)
class TranslatedSql:
    sql: str
    kind: str = "sql"
    db_name: Optional[str] = None
    table_name: Optional[str] = None


def strip_sql(action: str) -> str:
    sql = (action or "").strip()
    if sql.endswith(";"):
        sql = sql[:-1].strip()
    return sql


def normalize_identifier(identifier: str) -> str:
    identifier = strip_sql(identifier)
    identifier = identifier.strip("`\"[]")
    if "." in identifier:
        identifier = identifier.split(".")[-1].strip("`\"[]")
    return identifier


def quote_sqlite_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def translate_mysql_probe(action: str) -> TranslatedSql:
    sql = strip_sql(action)

    match = re.fullmatch(r"(?is)USE\s+([`\"\[]?[\w.-]+[`\"\]]?)", sql)
    if match:
        return TranslatedSql(sql="", kind="use", db_name=normalize_identifier(match.group(1)))

    match = re.fullmatch(r"(?is)SHOW\s+TABLES(?:\s+FROM\s+([`\"\[]?[\w.-]+[`\"\]]?))?", sql)
    if match:
        db_name = normalize_identifier(match.group(1)) if match.group(1) else None
        return TranslatedSql(
            sql=(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ),
            kind="show_tables",
            db_name=db_name,
        )

    match = re.fullmatch(r"(?is)(?:DESC|DESCRIBE)\s+([`\"\[]?[\w.-]+[`\"\]]?)", sql)
    if match:
        table = normalize_identifier(match.group(1))
        return TranslatedSql(
            sql=f"PRAGMA table_info({quote_sqlite_string(table)})",
            kind="describe",
            table_name=table,
        )

    match = re.fullmatch(r"(?is)SHOW\s+COLUMNS\s+FROM\s+([`\"\[]?[\w.-]+[`\"\]]?)", sql)
    if match:
        table = normalize_identifier(match.group(1))
        return TranslatedSql(
            sql=f"PRAGMA table_info({quote_sqlite_string(table)})",
            kind="describe",
            table_name=table,
        )

    return TranslatedSql(sql=translate_mysql_sql(sql), kind="sql")


def translate_mysql_sql(sql: str) -> str:
    sql = strip_sql(sql)
    sql = re.sub(r"`([^`]+)`", r'"\1"', sql)
    sql = re.sub(r"(?is)\bTRUE\b", "1", sql)
    sql = re.sub(r"(?is)\bFALSE\b", "0", sql)
    sql = re.sub(r"(?is)\bCURDATE\(\)", "date('now')", sql)
    sql = re.sub(r"(?is)\bNOW\(\)", "datetime('now')", sql)
    sql = re.sub(r"(?is)\bIFNULL\s*\(", "COALESCE(", sql)
    return sql


def sqlite_type_to_mysql(declared_type: Any) -> str:
    raw = str(declared_type or "").strip()
    if not raw:
        return "text"
    base = re.split(r"[\s(]", raw, maxsplit=1)[0].upper()
    mapped = SQLITE_TO_MYSQL_TYPES.get(base)
    if mapped:
        suffix = raw[len(base) :] if raw.upper().startswith(base) else ""
        return mapped + suffix
    return raw.lower()


def format_description_rows(rows: Sequence[Tuple[Any, ...]]) -> List[Tuple[Any, ...]]:
    out = []
    for row in rows:
        _cid, name, declared_type, not_null, default_value, pk = row
        mysql_type = sqlite_type_to_mysql(declared_type)
        key = "PRI" if pk else ""
        null = "NO" if not_null or pk else "YES"
        extra = "auto_increment" if pk and mysql_type == "int" else ""
        out.append((name, mysql_type, null, key, default_value, extra))
    return out
