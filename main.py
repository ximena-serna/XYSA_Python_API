# =========================
# Data Cleaning API — fully commented
# =========================

# Import FastAPI tools.
from fastapi import FastAPI, Body  # FastAPI lets us build web APIs; Body reads JSON from the request.

# Import typing helpers for type hints (purely for readability/help from editors).
from typing import Any, Dict, List, Optional  # These describe shapes of data (not required at runtime).

# Import Polars, a fast DataFrame library (similar to pandas but often faster).
import polars as pl  # We'll use 'pl' to refer to Polars.

# Import hashlib to create SHA-256 hashes (unique IDs for rows).
import hashlib  # Used to generate a stable, short ID from the content of each row.

# Create the FastAPI app instance with a descriptive title.
app = FastAPI(title="Data Cleaning API")  # This is the web server object.


# ---- Helpers ----

def make_hash_expr() -> pl.Expr:
    """
    Build a Polars expression that:
      1) Packs all columns into a struct (single "object" per row),
      2) Maps each row through a Python function that returns a short SHA-256 hash,
      3) Names the result column 'hash_id'.
    This returns a *lazy expression*, not the actual values; Polars will compute it when applied.
    """
    return (
        pl.struct(pl.all())  # Take all columns and bundle them into one struct per row.
        .map_elements(lambda s: _hash_row_dict(s))  # For each row-struct, call our Python function to hash it.
        .alias("hash_id")  # Name the resulting column 'hash_id'.
    )


def _hash_row_dict(s: dict) -> str:
    """
    Turn a row (as a dict) into a *canonical* string and hash it.
    - We sort keys to ensure the same ordering every time (so the same row => same hash).
    - We 'repr' the dict to a string, encode it as UTF-8, and SHA-256 hash it.
    - We keep only the first 12 hex characters (short but very unlikely to collide for small datasets).
    """
    canon = {k: s[k] for k in sorted(s.keys())}  # Sort keys so ordering is predictable.
    raw = repr(canon)  # Convert to a stable string representation.
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]  # 12-char prefix of the SHA-256 hex digest.


def trim_all_utf8(df: pl.DataFrame) -> pl.DataFrame:
    """
    Trim (remove leading/trailing spaces) on all text (Utf8) columns.
    Uses .str.strip_chars()
    - If there are no text columns, just return the DataFrame unchanged.
    """
    utf8_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt == pl.Utf8]  # Find columns with text type.
    if not utf8_cols:  # If none, do nothing.
        return df
    return df.with_columns([  # Otherwise, create trimmed versions:
        pl.col(c).str.strip_chars().alias(c)  # trim spaces on each text column.
        for c in utf8_cols
    ])


def normalize_date_simple(df: pl.DataFrame, date_fields: Optional[List[str]]) -> pl.DataFrame:
    """
    Convert columns listed in 'date_fields' from format 'MM-DD-YYYY HH:MM:SS' to 'YYYY/MM/DD' (string).
    Strategy (100% compatible with Polars 1.33.1):
      - Force the column to text (Utf8),
      - Trim spaces,
      - Parse using a fixed datetime format (strict=False means non-matching values become null),
      - Format back to a date string 'YYYY/MM/DD'.
    If 'date_fields' is None or empty, no changes are made.
    """
    if not date_fields:  # No date fields provided -> return as-is.
        return df

    exprs: List[pl.Expr] = []  # We'll collect per-column expressions here.

    for c in date_fields:
        if c not in df.columns:  # Skip silently if the column doesn't exist.
            continue
        exprs.append(
            pl.col(c)
            .cast(pl.Utf8, strict=False)  # Ensure the column is text.
            .str.strip_chars()  # Trim spaces.
            .str.strptime(pl.Datetime, format="%m-%d-%Y %H:%M:%S", strict=False)  # Try to parse the datetime.
            .dt.strftime("%Y/%m/%d")  # Convert to 'YYYY/MM/DD' string.
            .alias(c)  # Keep the original column name.
        )

    return df.with_columns(exprs) if exprs else df  # Apply all conversions (or return original if none).


def get_numeric_cols(df: pl.DataFrame, exclude: List[str]) -> List[str]:
    """
    Return the list of numeric columns, excluding:
      - any columns in the 'exclude' list (e.g., group-by keys),
      - the helper column 'hash_id'.
    Polars 1.33.1 compatibility:
      - Try the shortcut pl.NUMERIC_DTYPES, else fall back to a manual type set.
    """
    exclude_set = set(exclude) | {"hash_id"}  # Build a set of columns we must not aggregate.

    try:
        cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns  # Preferred: ask Polars for numeric dtypes.
    except Exception:
        # Fallback: define numeric types manually (covers common ints and floats).
        numeric_types = {
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64,
        }
        cols = [c for c, dt in zip(df.columns, df.dtypes) if dt in numeric_types]

    return [c for c in cols if c not in exclude_set]  # Remove excluded columns from the result.


def _groupby(df: pl.DataFrame, keys: List[str]):
    """
    Compatibility wrapper:
    - Some Polars versions use 'groupby', others 'group_by'. We try both.
    - Returns a groupby object that we can call .agg(...) on.
    """
    fn = getattr(df, "group_by", None) or getattr(df, "groupby")  # Pick whichever exists.
    return fn(keys)  # Return the groupby object.


# ---- Endpoint ----

# Example request body to show in the OpenAPI docs (FastAPI's automatic docs).
ExampleBody = {
    "data": [
        {"nombre": "  Ana  ", "categoria": "A", "fecha": "01-02-2025 10:30:00", "monto": 100},
        {"nombre": "Ana", "categoria": "A", "fecha": "01-02-2025 10:30:00", "monto": 100},
        {"nombre": " Luis ", "categoria": "B", "fecha": "01-31-2025 18:00:00", "monto": 50},
        {"nombre": "Pedro ", "categoria": "B", "fecha": "01-31-2025 20:00:00", "monto": 70}
    ],
    "group_by": ["categoria"],
    "date_fields": ["fecha"]
}


@app.post("/clean")
def clean_data(payload: Dict[str, Any] = Body(..., example=ExampleBody)):
    """
    Main endpoint:
      - Receives JSON with:
          data: list of row dicts,
          group_by: optional list of column names to group by,
          date_fields: optional list of date columns to normalize.
      - Returns a cleaned dataset with:
          * trimmed text,
          * normalized dates (if requested),
          * 'hash_id' per row,
          * duplicates removed (by 'hash_id'),
          * optional group aggregations (count + sums of numeric columns),
          * sample of first 3 'hash_id' values.
    """
    # Pull fields out of the incoming JSON payload, with safe defaults.
    data: List[Dict[str, Any]] = payload.get("data", [])  # List of rows (each row is a dict).
    group_by: List[str] = payload.get("group_by", []) or []  # Columns to group by (optional).
    date_fields: Optional[List[str]] = payload.get("date_fields")  # Columns to normalize as dates (optional).

    # Validate 'data' quickly: it must be a non-empty list.
    if not isinstance(data, list) or not data:
        return {"error": "El campo 'data' debe ser una lista con al menos 1 objeto."}  # Simple error message.

    # 1) Convert JSON rows -> Polars DataFrame.
    df = pl.DataFrame(data, strict=False)  # strict=False allows mixed/nullable inputs without crashing.

    # 2) Trim whitespace on all Utf8 (text) columns.
    df = trim_all_utf8(df)  # Clean text columns so comparisons/hashes are consistent.

    # 3) Normalize date columns from 'MM-DD-YYYY HH:MM:SS' -> 'YYYY/MM/DD' (string).
    df = normalize_date_simple(df, date_fields)  # Non-matching values become null (by design).

    # 4) Add a 'hash_id' column based on the entire row's contents.
    df = df.with_columns([make_hash_expr()])  # Stable short ID helps deduplicate rows.

    # 5) Deduplicate by 'hash_id'.
    before = df.height  # Remember original row count.
    df = df.unique(subset=["hash_id"], keep="first")  # Keep the first occurrence of each unique hash.
    removed = before - df.height  # Calculate how many duplicates were removed.

    # 6) Optional group-by aggregation (if group_by was provided and valid).
    groups: List[Dict[str, Any]] = []  # We'll store group results as a list of dicts.
    if group_by:
        keys = [k for k in group_by if k in df.columns]  # Only keep keys that actually exist in the DataFrame.
        if keys:
            numeric_cols = get_numeric_cols(df, exclude=keys)  # Numeric columns to aggregate (exclude keys/hash).
            # Build aggregations:
            #  - pl.len(): row count per group (named 'count')
            #  - sum of each numeric column (cast to Float64 to avoid integer overflow / dtype issues)
            aggs = [pl.len().alias("count")] + [
                pl.col(c).cast(pl.Float64, strict=False).sum().alias(f"sum_{c}")
                for c in numeric_cols
            ]
            gdf = _groupby(df, keys).agg(aggs)  # Perform the groupby and aggregations.
            groups = []
            for row in gdf.to_dicts():  # Convert grouped result to a list of dicts.
                group_keys = {k: row.pop(k) for k in keys}  # Extract the grouping key values.
                groups.append({**group_keys, **row})  # Merge keys with the aggregated metrics.

    # Return a JSON-friendly dict with metadata and the cleaned data.
    return {
        "rows_in": before,  # How many rows came in.
        "rows_out": df.height,  # How many rows after cleaning/dedup.
        "duplicates_removed": removed,  # Number of duplicates removed.
        "groups": groups,  # Group-by results (empty if none requested).
        "data": df.to_dicts(),  # Full cleaned rows as a list of dicts.
    }