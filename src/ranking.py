import pandas as pd 

class EmployeeScoring:
    """
    Task 3 — Compute monthly sentiment scores per employee.

    Maps categorical sentiments to numeric values and aggregates them by employee
    and month to produce a table suitable for downstream ranking and risk analysis.

    Notes
    -----
    - Per project spec, message scores are: Positive=+1, Negative=-1, Neutral=0.
    - Per project spec, aggregation is by calendar month with scores *summed* per
      employee per month (the "score resets" each new month).
    - This implementation currently takes the mean by day, converts to monthly
      periods, then sums those means. If you want strict compliance with the spec,
      aggregate *sum* directly over calendar months.

    Attributes
    ----------
    df : pandas.DataFrame
        Working copy of the input with an added 'sentiment_num' column.
    rankings : pandas.DataFrame
        Last computed monthly aggregation (returned by `compute_scores`).

    Required Input Columns on `df`
    ------------------------------
    - 'employee_id' : hashable employee key.
    - 'date'        : datetime64[ns] or string parseable to datetime.
    - 'sentiment'   : one of {'POSITIVE','NEGATIVE','NEUTRAL'} (case-insensitive).

    Output Columns
    --------------
    - 'employee_id' : employee key.
    - 'date'        : pandas.Period[M] representing the month.
    - 'sentiment_num' : aggregated numeric sentiment for that employee-month.
    """

    global SENT_MAP
    SENT_MAP = {
        "POSITIVE": 1, "Positive": 1,
        "NEGATIVE": -1, "Negative": -1,
        "NEUTRAL": 0,  "Neutral": 0
    }

    def __init__(self, df):
        """
        Parameters
        ----------
        df : pandas.DataFrame
            Raw messages with at least ['employee_id','date','sentiment'].

        Side Effects
        ------------
        - Copies `df` to avoid mutating caller data.
        - Adds numeric column 'sentiment_num' via SENT_MAP.

        Raises
        ------
        KeyError
            If any required columns are missing.
        """
        self.df = df.copy()
        self.df["sentiment_num"] = self.df["sentiment"].map(SENT_MAP)
        self.rankings = pd.DataFrame()

    def compute_scores(self) -> pd.DataFrame:
        """
        Aggregate numeric sentiment by employee and month.

        Returns
        -------
        pandas.DataFrame
            A table with monthly sentiment per employee.

        Current Logic
        -------------
        1) Group by ['employee_id','date'] and take mean of 'sentiment_num'.
        2) Convert 'date' to monthly Period (M).
        3) Group again by ['employee_id','date'] and sum.

        Spec-Accurate Alternative
        -------------------------
        To strictly match “sum of messages per calendar month”:
            df2 = self.df.copy()
            df2["month"] = pd.to_datetime(df2["date"]).dt.to_period("M")
            out = (df2.groupby(["employee_id","month"], as_index=False)["sentiment_num"]
                        .sum()
                        .rename(columns={"month":"date"}))
            return out

        Notes
        -----
        - Ensures output 'date' is monthly Period[M] to keep month semantics.
        - Keeps column name 'date' for compatibility with downstream code.

        Examples
        --------
        >>> scorer = EmployeeScoring(df)
        >>> monthly = scorer.compute_scores()
        >>> monthly.head()
            employee_id    date  sentiment_num
        0           E01  2025-07              3
        1           E02  2025-07             -1
        """
        self.rankings = self.df.groupby(
            ["employee_id", "date"], as_index=False
        )["sentiment_num"].mean()

        self.rankings["date"] = self.rankings["date"].dt.to_period("M")

        self.rankings = self.rankings.groupby(
            ["employee_id", "date"], as_index=False
        ).sum()

        return self.rankings


class EmployeeRanking:
    """
    Task 4 — Rank employees monthly based on sentiment scores.

    Produces the top 3 and bottom 3 employees per month using the numeric
    sentiment produced by `EmployeeScoring.compute_scores`.

    Parameters
    ----------
    df : pandas.DataFrame
        If `scores_available=False`, this is raw messages with
        ['employee_id','date','sentiment'] so that monthly scores can be computed.
        If `scores_available=True`, this must already be the result of
        `EmployeeScoring(...).compute_scores()` with columns:
          - 'employee_id' (key)
          - 'date' (Period[M] or month-like)
          - 'sentiment_num' (numeric monthly score)
    scores_available : bool, default False
        Toggle for whether `df` already contains monthly scores.

    Attributes
    ----------
    df : pandas.DataFrame
        Monthly scores table used for ranking.
    rankings : pandas.DataFrame
        Last produced rankings table.

    Ranking Semantics
    -----------------
    - Positive rankings: highest 'sentiment_num' first (desc), then employee_id asc.
    - Negative rankings: lowest 'sentiment_num' first (asc), then employee_id asc.
    - Top-k is k=3 per month by default via `groupby(...).head(3)`.

    Notes
    -----
    - Ties break by employee_id ascending.
    - Sorting includes 'date' to keep month blocks together.
    - All `get_*` methods update `self.rankings`.
    """

    def __init__(self, df, scores_available=False):
        """
        See class docstring.

        Side Effects
        ------------
        - If `scores_available=False`, computes monthly scores from raw messages.

        Raises
        ------
        KeyError
            If required columns are missing for the chosen mode.
        """
        if not scores_available:
            self.df = EmployeeScoring(df).compute_scores()
        else:
            self.df = df.copy()
        self.rankings = pd.DataFrame()

    def get_positive_rankings(self) -> pd.DataFrame:
        """
        Return the top 3 employees per month with the highest sentiment scores.

        Returns
        -------
        pandas.DataFrame
            Subset of `self.df` with at most 3 rows per month.

        Sorting
        -------
        - by=["date","sentiment_num","employee_id"]
        - ascending=[True, False, True]

        Examples
        --------
        >>> ranker = EmployeeRanking(monthly_df, scores_available=True)
        >>> top3 = ranker.get_positive_rankings()
        """
        self.rankings = self.df.sort_values(
            by=["date","sentiment_num","employee_id"],
            ascending=[True, False, True]
        ).groupby("date", as_index=False).head(3)
        return self.rankings

    def get_negative_rankings(self) -> pd.DataFrame:
        """
        Return the top 3 employees per month with the lowest sentiment scores.

        Returns
        -------
        pandas.DataFrame
            Subset of `self.df` with at most 3 rows per month.

        Sorting
        -------
        - by=["date","sentiment_num","employee_id"]
        - ascending=[True, True, True]

        Examples
        --------
        >>> ranker = EmployeeRanking(monthly_df, scores_available=True)
        >>> bottom3 = ranker.get_negative_rankings()
        """
        self.rankings = self.df.sort_values(
            by=["date","sentiment_num","employee_id"],
            ascending=[True, True, True]
        ).groupby("date", as_index=False).head(3)
        return self.rankings

    def get_rankings(self, drop_type: bool = False) -> pd.DataFrame:
        """
        Produce a combined table of bottom-3 and top-3 per month.

        Parameters
        ----------
        drop_type : bool, default False
            If True, drop the 'Type' column before returning.

        Returns
        -------
        pandas.DataFrame
            Concatenated rankings with a 'Type' column in {"Negative","Positive"}.

        Sorting
        -------
        by=["date","Type","sentiment_num","employee_id"],
        ascending=[True, False, False, True]
        - Puts "Positive" after "Negative" when Type is sorted descending,
          and orders by score accordingly.

        Examples
        --------
        >>> ranker = EmployeeRanking(monthly_df, scores_available=True)
        >>> both = ranker.get_rankings()
        >>> both.head()
             employee_id    date  sentiment_num     Type
        0            E04  2025-07            -3  Negative
        1            E12  2025-07            -2  Negative
        2            E02  2025-07            -1  Negative
        3            E01  2025-07             3  Positive
        4            E05  2025-07             2  Positive
        """
        negative_rankings = self.get_negative_rankings().copy()
        negative_rankings["Type"] = "Negative"

        positive_rankings = self.get_positive_rankings().copy()
        positive_rankings["Type"] = "Positive"

        grouped_rankings = pd.concat(
            [negative_rankings, positive_rankings], ignore_index=True
        )

        grouped_rankings.sort_values(
            by=["date","Type","sentiment_num","employee_id"],
            ascending=[True, False, False, True],
            inplace=True
        )

        if drop_type:
            grouped_rankings.drop(columns=["Type"], inplace=True)

        return grouped_rankings
