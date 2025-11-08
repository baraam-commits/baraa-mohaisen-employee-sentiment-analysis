import pandas as pd
import re
from pathlib import Path
from typing import Optional, Dict

class LoadData:
    """
    Task 0 — Data ingestion and normalization.

    Supports two schemas:
      RAW:          ['Subject','body','from','date']
      PREPROCESSED: includes at least ['employee_id','date'] and may have ['sentiment_num','text',...]

    Output (both cases):
      - 'employee_id' : lowercased sender id (raw only; pass-through for preprocessed)
      - 'date'        : pandas datetime64[ns]
      - 'month'       : pandas Period[M]
      - 'text'        : Subject + body normalized (raw only; preserved if present)
      - 'text_len'    : len(text)
      - 'word_count'  : count of word tokens in text (letters+digits)
      - All original columns retained
    """

    RAW_REQ  = {"Subject", "body", "from", "date"}
    PRE_REQ  = {"employee_id", "date"}

    def __init__(self, file_path: Optional[str] = None, columns_map: Optional[Dict[str, str]] = None):
        """
        columns_map: optional rename dict before processing, e.g. {'From':'from','Date':'date'}
        """
        self.file_path = file_path
        self.columns_map = columns_map or {}
        self.df = pd.DataFrame()

    # ---------------- internal utils ----------------
    @staticmethod
    def _norm_text(s: pd.Series) -> pd.Series:
        # collapse whitespace, strip control chars
        return (
            s.fillna("")
             .astype(str)
             .str.replace(r"\s+", " ", regex=True)
             .str.strip()
        )

    @staticmethod
    def _extract_employee_id(from_col: pd.Series) -> pd.Series:
        # take substring before '@', lowercased
        return (
            from_col.fillna("")
                    .astype(str)
                    .str.extract(r"([^@]+)", expand=False)
                    .str.lower()
                    .fillna("")
        )

    @staticmethod
    def _add_lengths(df: pd.DataFrame) -> pd.DataFrame:
        # derive text_len and word_count if 'text' exists
        if "text" in df.columns:
            df["text_len"] = df["text"].str.len()
            # alphanum word tokens
            df["word_count"] = df["text"].str.findall(r"\b[0-9A-Za-z]+\b").str.len()
        else:
            # safe defaults
            df["text_len"] = 0
            df["word_count"] = 0
        return df

    @staticmethod
    def _ensure_datetime(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        df = df.dropna(subset=[date_col])
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        return df

    # ---------------- schema handlers ----------------
    def _load_raw(self, df: pd.DataFrame, clean: bool) -> pd.DataFrame:
        # normalize text if clean, else just build minimal columns
        if clean:
            text = self._norm_text(df["Subject"]) + " " + self._norm_text(df["body"])
            df["text"] = self._norm_text(text)
            df["employee_id"] = self._extract_employee_id(df["from"])
            # drop duplicates on key tuple
            df = df.drop_duplicates(subset=["employee_id", "date", "text"])
        else:
            # minimal: ensure required cols, derive employee_id if missing
            if "employee_id" not in df.columns:
                df["employee_id"] = self._extract_employee_id(df["from"])
            if "text" not in df.columns:
                text = df["Subject"].fillna("").astype(str) + " " + df["body"].fillna("").astype(str)
                df["text"] = text

        df = self._ensure_datetime(df, "date")
        df = df.sort_values(["employee_id", "date"], kind="mergesort").reset_index(drop=True)
        df["month"] = df["date"].dt.to_period("M")
        df = self._add_lengths(df)
        return df

    def _load_preprocessed(self, df: pd.DataFrame, clean: bool) -> pd.DataFrame:
        # keep as-is, just enforce datetime, order, month, lengths if text exists
        df = self._ensure_datetime(df, "date")
        # ensure column types
        df["employee_id"] = df["employee_id"].astype(str).str.lower()
        df = df.sort_values(["employee_id", "date"], kind="mergesort").reset_index(drop=True)
        df["month"] = df["date"].dt.to_period("M")
        if "text" in df.columns and clean:
            df["text"] = self._norm_text(df["text"])
        df = self._add_lengths(df)
        return df

    # ---------------- public API ----------------
    def load_pandas_dataframe(self, clean: bool = True, file_path: Optional[str] = None) -> pd.DataFrame:
        path = Path(file_path or self.file_path)
        if not path:
            raise FileNotFoundError("No file_path provided.")
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

        df = pd.read_csv(path)
        if self.columns_map:
            df = df.rename(columns=self.columns_map)

        cols = set(df.columns)

        # choose handler
        if self.RAW_REQ.issubset(cols):
            self.df = self._load_raw(df, clean=clean)
        elif self.PRE_REQ.issubset(cols):
            self.df = self._load_preprocessed(df, clean=clean)
        else:
            missing_raw = self.RAW_REQ - cols
            missing_pre = self.PRE_REQ - cols
            raise KeyError(
                "CSV schema not recognized.\n"
                f"- Missing RAW columns: {missing_raw}\n"
                f"- Missing PREPROCESSED columns: {missing_pre}"
            )
        return self.df
