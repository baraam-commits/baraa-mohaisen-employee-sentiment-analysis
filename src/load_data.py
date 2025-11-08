import pandas as pd
import csv

class LoadData:
    """
    Task 0 — Data ingestion and normalization.

    Reads a raw CSV export of employee communications and prepares it
    for sentiment analysis and downstream tasks (EDA, scoring, ranking).

    Expected Input CSV Columns
    --------------------------
    - 'body'    : message text body
    - 'Subject' : message subject line
    - 'from'    : sender email address
    - 'date'    : send timestamp (any parseable string format)

    Parameters
    ----------
    file_path : str or PathLike
        Path to the CSV file to load.

    Attributes
    ----------
    file_path : str
        Location of the source CSV file on disk.

    Output DataFrame Columns
    -------------------------
    - 'text'         : concatenated subject + body, normalized
    - 'employee_id'  : lowercase string before '@' in sender address
    - 'date'         : datetime64[ns] object
    - All original columns retained unless dropped during cleaning

    Notes
    -----
    - Cleaning removes duplicates by ['employee_id','date','text'].
    - Non-printable and excessive whitespace are stripped from text.
    - Sorting by ['from','date'] ensures chronological order per employee.
    - When `clean=False`, skips heavy text normalization to speed up reloads.
    """

    def __init__(self, file_path: str):
        """
        Initialize with file path.

        Parameters
        ----------
        file_path : str
            Path to the CSV dataset.
        """
        self.file_path = file_path

    def load_pandas_dataframe(self, clean: bool = True) -> pd.DataFrame:
        """
        Load and optionally clean the employee message dataset.

        Parameters
        ----------
        clean : bool, default True
            Whether to perform full normalization.  
            Set to False when the CSV has already been preprocessed
            (e.g., during reload after sentiment labeling).

        Returns
        -------
        pandas.DataFrame
            Normalized DataFrame with the following added/processed columns:
              - 'text' : combined and cleaned message text
              - 'employee_id' : lowercase sender identifier
              - 'date' : converted to pandas datetime, sorted by employee

        Processing Steps
        ----------------
        1) Read CSV with pandas.
        2) If `clean=True`:
             - Concatenate 'Subject' + 'body' → 'text'.
             - Extract 'employee_id' from 'from' (before '@').
             - Drop duplicates by ['employee_id','date','text'].
             - Strip newlines, multiple spaces, and non-printables.
           Else:
             - Only drop rows with null 'date'.
        3) Convert 'date' to datetime64.
        4) Sort by ['from','date'] for stable chronological order.
        5) Return cleaned DataFrame.

        Raises
        ------
        FileNotFoundError
            If the CSV path is invalid.
        KeyError
            If required columns ('body','from','date','Subject') are missing.

        Examples
        --------
        >>> loader = LoadData("employee_emails.csv")
        >>> df = loader.load_pandas_dataframe(clean=True)
        >>> df.head()
               from        date        employee_id        text
        0   jsmith@... 2025-01-03  jsmith  Project update looks good.
        1   jdoe@...   2025-01-04  jdoe    Scheduling next meeting...
        """
        data_frame = pd.read_csv(self.file_path)

        def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
            if ["Subject","body","date","from"] in df.columns:
                
                if clean:
                    # Combine subject + body into normalized text
                    df["text"] = (df["Subject"].fillna("") + " " + df["body"].fillna("")).str.strip()

                    # Extract employee_id before '@'
                    df["employee_id"] = df["from"].str.extract(r"([^@]+)").iloc[:, 0].str.lower()

                    # Drop duplicates by key fields
                    df = df.drop_duplicates(subset=["employee_id", "date", "text"])

                    # Normalize whitespace and strip non-printables
                    import re
                    df["text"] = df["text"].apply(
                        lambda s: re.sub(r"\s+", " ", s.strip()) if isinstance(s, str) else ""
                    )
                else:
                    # Only ensure date column is valid
                    df = df.dropna(subset=["date"])

                # Convert and sort by sender/date for temporal analysis
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values(["from", "date"], kind="mergesort").reset_index(drop=True)
                return df
            elif ["employee_id","date","sentiment_num"] in df.columns:
                return df.sort_values(by = "employee_id", inplace= True)
        return _prep_df(data_frame)
