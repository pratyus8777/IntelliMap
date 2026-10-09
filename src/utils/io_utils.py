from pathlib import Path
import json
import re
from typing import Dict, List, Tuple, Any
import pandas as pd

DDL_TABLE_RE = re.compile(r"CREATE\s+TABLE\s+([A-Za-z0-9_]+)\s*\(", re.IGNORECASE)
DDL_COL_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s+([A-Za-z0-9_()]+)", re.IGNORECASE)

def read_json(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)

def read_csv_safe(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        # return empty DataFrame but do not raise (downstream will handle)
        return pd.DataFrame()
    try:
        return pd.read_csv(p, nrows=nrows, low_memory=False, encoding_errors="ignore")
    except Exception as e:
        print(f"[WARN] Failed to read CSV {p}: {e}")
        return pd.DataFrame()

def load_csvs(root: str | Path, files_map: Dict[str, str], nrows: int | None = 500) -> Dict[str, pd.DataFrame]:
    root = Path(root)
    out: Dict[str, pd.DataFrame] = {}
    for logical, fname in (files_map or {}).items():
        p = root / fname
        df = read_csv_safe(p, nrows=nrows)
        out[logical] = df
    return out

def parse_ddl(path: str | Path) -> Dict[str, Dict[str, str]]:
    """
    Parse a DDL.sql and return {TableName: {column_name: datatype, ...}, ...}
    Best-effort, tolerant to formatting.
    """
    p = Path(path)
    if not p.exists():
        print(f"[WARN] DDL file not found: {p}")
        return {}
    text = p.read_text(encoding="utf-8", errors="ignore")
    schema: Dict[str, Dict[str, str]] = {}
    # naive split by CREATE TABLE
    parts = re.split(r"CREATE\s+TABLE", text, flags=re.IGNORECASE)
    for part in parts[1:]:
        try:
            header_rest = part.strip()
            # get table name
            m = re.match(r"\s*([A-Za-z0-9_]+)", header_rest)
            if not m:
                continue
            table = m.group(1)
            schema[table] = {}
            # body between first "(" and corresponding ");"
            body_start = part.find("(")
            body_end = part.find(");", body_start)
            body = part[body_start+1:body_end] if body_start >= 0 and body_end > body_start else ""
            lines = [l.strip().rstrip(",") for l in body.splitlines() if l.strip()]
            for line in lines:
                if line.upper().startswith(("PRIMARY KEY","FOREIGN KEY","CONSTRAINT","UNIQUE")):
                    continue
                cm = DDL_COL_RE.match(line)
                if cm:
                    col = cm.group(1)
                    dtype = cm.group(2)
                    schema[table][col] = dtype
        except Exception as e:
            print(f"[WARN] Failed parsing table in DDL: {e}")
            continue
    return schema
