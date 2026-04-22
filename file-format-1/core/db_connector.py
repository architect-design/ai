"""
Database Connector
==================
Connects to Oracle, PostgreSQL, MySQL, SQLite, or any
SQLAlchemy-supported database to read source data for test-data generation.

Usage:
    connector = DBConnector()
    connector.connect("oracle+cx_Oracle://user:pass@host:1521/?service_name=orcl")
    rows = connector.fetch("SELECT * FROM GL_JOURNAL_LINES WHERE ROWNUM <= 10")
    connector.disconnect()
"""

import re
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# SQLAlchemy is optional – gracefully degrade if not installed
try:
    from sqlalchemy import create_engine, text, inspect
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

# cx_Oracle / oracledb optional
try:
    import oracledb
    ORACLE_AVAILABLE = True
except ImportError:
    try:
        import cx_Oracle as oracledb
        ORACLE_AVAILABLE = True
    except ImportError:
        ORACLE_AVAILABLE = False


class DBConnectionError(Exception):
    pass


class DBConnector:
    """
    Universal database connector.

    Supported connection string formats:
      sqlite:///path/to/db.sqlite3
      postgresql://user:pass@host:5432/dbname
      mysql+pymysql://user:pass@host:3306/dbname
      oracle+oracledb://user:pass@host:1521/?service_name=ORCL
      oracle+cx_Oracle://user:pass@host:1521/SID
    """

    PRESETS = {
        "oracle_sample": {
            "driver":      "oracle+oracledb",
            "description": "Oracle DB (oracledb driver)",
            "placeholder": "oracle+oracledb://user:pass@host:1521/?service_name=ORCL",
        },
        "oracle_cx": {
            "driver":      "oracle+cx_Oracle",
            "description": "Oracle DB (cx_Oracle legacy driver)",
            "placeholder": "oracle+cx_Oracle://user:pass@host:1521/SID",
        },
        "postgres": {
            "driver":      "postgresql",
            "description": "PostgreSQL",
            "placeholder": "postgresql://user:pass@localhost:5432/mydb",
        },
        "mysql": {
            "driver":      "mysql+pymysql",
            "description": "MySQL / MariaDB",
            "placeholder": "mysql+pymysql://user:pass@localhost:3306/mydb",
        },
        "sqlite": {
            "driver":      "sqlite",
            "description": "SQLite (local file)",
            "placeholder": "sqlite:///mydb.sqlite3",
        },
        "sqlserver": {
            "driver":      "mssql+pyodbc",
            "description": "SQL Server",
            "placeholder": "mssql+pyodbc://user:pass@host/db?driver=ODBC+Driver+17+for+SQL+Server",
        },
    }

    def __init__(self):
        self._engine = None
        self._conn   = None
        self.conn_str: str | None = None
        self.connected: bool = False

    # ── Lifecycle ─────────────────────────────────────────────────────
    def connect(self, connection_string: str) -> bool:
        """Establish a database connection. Returns True on success."""
        if not SQLALCHEMY_AVAILABLE:
            raise DBConnectionError(
                "SQLAlchemy is not installed. Run: pip install sqlalchemy"
            )
        try:
            self._engine  = create_engine(connection_string, pool_pre_ping=True)
            self._conn    = self._engine.connect()
            self.conn_str = self._mask_password(connection_string)
            self.connected = True
            logger.info("Connected to: %s", self.conn_str)
            return True
        except Exception as exc:
            self.connected = False
            raise DBConnectionError(f"Connection failed: {exc}") from exc

    def disconnect(self):
        try:
            if self._conn:
                self._conn.close()
            if self._engine:
                self._engine.dispose()
        except Exception:
            pass
        finally:
            self._conn     = None
            self._engine   = None
            self.connected = False

    # ── Data retrieval ────────────────────────────────────────────────
    def fetch(self, query: str, params: dict | None = None,
              max_rows: int = 1000) -> list[dict]:
        """
        Execute a SELECT query and return rows as list-of-dicts.
        Automatically limits to max_rows.
        """
        self._assert_connected()
        safe_query = self._add_row_limit(query, max_rows)
        try:
            result = self._conn.execute(text(safe_query), params or {})
            columns = list(result.keys())
            rows = []
            for row in result:
                rows.append({col: self._serialize(val)
                             for col, val in zip(columns, row)})
            return rows
        except Exception as exc:
            raise DBConnectionError(f"Query failed: {exc}") from exc

    def list_tables(self) -> list[str]:
        """Return list of all table names in the current schema."""
        self._assert_connected()
        try:
            inspector = inspect(self._engine)
            return inspector.get_table_names()
        except Exception as exc:
            raise DBConnectionError(f"Could not list tables: {exc}") from exc

    def describe_table(self, table_name: str) -> list[dict]:
        """Return column definitions for a table."""
        self._assert_connected()
        try:
            inspector = inspect(self._engine)
            cols = inspector.get_columns(table_name)
            return [
                {"name": c["name"], "type": str(c["type"]), "nullable": c["nullable"]}
                for c in cols
            ]
        except Exception as exc:
            raise DBConnectionError(f"Could not describe table '{table_name}': {exc}") from exc

    def preview_table(self, table_name: str, limit: int = 10) -> list[dict]:
        """Quick preview of a table's first N rows."""
        safe = re.sub(r'[^\w.]', '', table_name)  # sanitise
        return self.fetch(f"SELECT * FROM {safe}", max_rows=limit)

    # ── Test / Mock mode ──────────────────────────────────────────────
    def get_mock_data(self, format_type: str, n: int = 10) -> list[dict]:
        """
        Return mock database rows shaped for a given format.
        Used when no real DB is available.
        """
        import random, string
        from datetime import datetime, timedelta

        rng = random.Random(42)

        def rand_str(k):
            return ''.join(rng.choices(string.ascii_uppercase + string.digits, k=k))

        def rand_date():
            d = datetime.now() + timedelta(days=rng.randint(-90, 0))
            return d.strftime('%Y-%m-%d')

        def rand_amount():
            return round(rng.uniform(10, 99999), 2)

        rows = []
        for i in range(n):
            if format_type == 'nacha':
                rows.append({
                    'INDIVIDUAL_NAME':          f"JOHN SMITH {i+1}",
                    'DFI_ACCOUNT_NUMBER':       str(rng.randint(10**8, 10**9-1)),
                    'AMOUNT':                   str(int(rand_amount() * 100)).zfill(10),
                    'RDFI_ROUTING_TRANSIT':     '02100002',
                    'CHECK_DIGIT':              '4',
                    'TRANSACTION_CODE':         rng.choice(['22', '27']),
                    'INDIVIDUAL_IDENTIFICATION_NUMBER': rand_str(10),
                })
            elif format_type == 'oracle_gl':
                rows.append({
                    'SEGMENT1':     str(rng.randint(1000, 1999)),
                    'SEGMENT2':     str(rng.randint(2000, 2999)),
                    'SEGMENT3':     str(rng.randint(3000, 4999)),
                    'ENTERED_DR':   str(rand_amount()),
                    'ENTERED_CR':   '0',
                    'EFFECTIVE_DATE': rand_date(),
                    'DESCRIPTION':  f"JOURNAL ENTRY {i+1}",
                    'STATUS':       'N',
                    'ACTUAL_FLAG':  'A',
                    'CURRENCY_CODE': 'USD',
                })
            elif format_type == 'visa_vcf':
                rows.append({
                    'ACCOUNT_NUMBER':    str(rng.randint(4000000000000000, 4999999999999999)),
                    'TRANSACTION_AMOUNT': str(int(rand_amount() * 100)).zfill(12),
                    'AUTHORIZATION_CODE': rand_str(6),
                    'MERCHANT_NAME':     rng.choice(['WALMART', 'TARGET', 'AMAZON']),
                    'MERCHANT_CITY':     rng.choice(['NEW YORK', 'LOS ANGELES', 'CHICAGO']),
                    'TRANSACTION_CODE':  rng.choice(['05', '06', '25']),
                })
            else:
                rows.append({f'FIELD_{j}': rand_str(8) for j in range(1, 6)})

        return rows

    # ── Helpers ───────────────────────────────────────────────────────
    def _assert_connected(self):
        if not self.connected or self._conn is None:
            raise DBConnectionError("Not connected. Call connect() first.")

    def _mask_password(self, cs: str) -> str:
        return re.sub(r'(:)[^:@]+(@)', r'\1****\2', cs)

    def _add_row_limit(self, query: str, limit: int) -> str:
        q = query.strip().rstrip(';')
        ql = q.lower()
        if 'rownum' in ql or 'fetch first' in ql or 'limit' in ql or 'top ' in ql:
            return q
        # Oracle style
        return f"SELECT * FROM ({q}) WHERE ROWNUM <= {limit}"

    def _serialize(self, val: Any) -> Any:
        """Convert DB types to JSON-serializable Python types."""
        if val is None:
            return None
        if hasattr(val, 'isoformat'):        # date/datetime
            return val.isoformat()
        if isinstance(val, (int, float, str, bool)):
            return val
        return str(val)

    @property
    def status(self) -> dict:
        return {
            "connected":   self.connected,
            "connection":  self.conn_str or "N/A",
            "sqlalchemy":  SQLALCHEMY_AVAILABLE,
            "oracle_driver": ORACLE_AVAILABLE,
        }
