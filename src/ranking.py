import pandas as pd 

class EmployeeScoring:
    """
    Task 3 — Compute monthly sentiment scores per employee; detect flight risk.

    Maps categorical sentiments to numeric values and aggregates them by employee
    and month. Also flags employees with sustained negative sentiment in rolling
    30-day windows.

    Notes
    -----
    - Sentiment mapping: Positive=+1, Neutral=0, Negative=−1.
    - Monthly aggregation in `compute_scores` currently averages per day, converts
      to Period[M], then sums. For spec-exact monthly sums, see method docstring.
    - `flight_risk_analysis` requires a datetime 'date' and evaluates rolling
      30-day sums within each employee.

    Attributes
    ----------
    df : pandas.DataFrame
        Input with at least ['employee_id','date','sentiment'] and the derived
        'sentiment_num' column added in __init__.

    Required Input Columns on `df`
    ------------------------------
    - 'employee_id' : hashable employee key
    - 'date'        : datetime64[ns] or parseable string (must be datetime for rolling)
    - 'sentiment'   : {'POSITIVE','NEGATIVE','NEUTRAL'} (case-insensitive)

    Output Columns (compute_scores)
    -------------------------------
    - 'employee_id' : employee key
    - 'date'        : pandas.Period[M]
    - 'sentiment_num' : aggregated monthly score
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
            Raw messages with ['employee_id','date','sentiment'].

        Side Effects
        ------------
        - Adds numeric 'sentiment_num' via SENT_MAP.

        Raises
        ------
        KeyError
            If required columns are missing.
        """
        self.df = df
        self.df["sentiment_num"] = self.df["sentiment"].map(SENT_MAP)

    def compute_scores(self) -> pd.DataFrame:
        """
        Aggregate numeric sentiment by employee and month.

        Returns
        -------
        pandas.DataFrame
            Monthly sentiment per employee with columns
            ['employee_id','date','sentiment_num'] where 'date' is Period[M].

        Current Logic
        -------------
        1) Group by ['employee_id','date'] and mean 'sentiment_num'.
        2) Convert 'date' to monthly Period (M).
        3) Group again by ['employee_id','date'] and sum.

        Spec-Accurate Alternative
        -------------------------
        To sum messages directly by calendar month:
            df2 = self.df.copy()
            df2["month"] = pd.to_datetime(df2["date"]).dt.to_period("M")
            out = (df2.groupby(["employee_id","month"], as_index=False)["sentiment_num"]
                        .sum()
                        .rename(columns={"month":"date"}))
            return out

        Notes
        -----
        - Keeps column name 'date' for downstream compatibility.

        Examples
        --------
        >>> scorer = EmployeeScoring(df)
        >>> monthly = scorer.compute_scores()
        """
        rankings = self.df.copy()
        rankings = rankings.groupby(["employee_id", "date"], as_index=False)["sentiment_num"].mean()
        rankings["date"] = rankings["date"].dt.to_period("M")
        rankings = rankings.groupby(["employee_id", "date"], as_index=False).sum()
        return rankings

    def flight_risk_analysis(self, return_names_only: bool = True, min_neg_in_30d: int = 4) -> pd.DataFrame:
        """
        Identify employees with sustained negative sentiment in the last 30 days.

        Flags "flight risk" when the rolling 30-day count of NEGATIVE messages
        for an employee is >= `min_neg_in_30d`.

        Parameters
        ----------
        return_names_only : bool, default True
            - True  → return unique ['employee_id'] row for employees that ever
                      meet the criterion.
            - False → return summary per employee with:
                      ['employee_id','first_flight_risk','max_neg_sent_in_30d'].
        min_neg_in_30d : int, default 4
            Threshold for the rolling 30-day negative count.

        Returns
        -------
        pandas.DataFrame
            If `return_names_only=True`:
                Columns: ['employee_id'] (unique list of at-risk employees)
            Else:
                Columns:
                  - 'employee_id'
                  - 'first_flight_risk' : earliest timestamp the condition is met
                  - 'max_neg_sent_in_30d' : peak rolling 30-day negative count

        Requirements
        ------------
        - 'date' must be dtype datetime64[ns].
        - Data are evaluated within each employee using a datetime index and
          `rolling("30D", closed="both")`. Sorting by 'date' is done inside
          the per-employee function.

        Method
        ------
        - Create indicator `neg = 1` if sentiment_num == −1 else 0.
        - For each employee:
            - set_index('date'); compute `neg_30d` via time-based rolling sum.
            - `flight_risk = neg_30d >= min_neg_in_30d`.
        - Return either unique employee ids or per-employee summaries.

        Examples
        --------
        >>> scorer = EmployeeScoring(df)  # df['date'] already to_datetime
        >>> risks = scorer.flight_risk_analysis()
        >>> risks_all = scorer.flight_risk_analysis(return_names_only=False, min_neg_in_30d=5)
        """
        flight_risks = self.df.copy()
        flight_risks["neg"] = (flight_risks["sentiment_num"] == -1).astype(int)

        def _within_emp(g: pd.DataFrame) -> pd.DataFrame:
            g = g.set_index("date")
            g["neg_30d"] = g["neg"].rolling("30D", closed="both").sum()
            g["flight_risk"] = g["neg_30d"] >= min_neg_in_30d
            return g.reset_index()

        flight_risks = (
            flight_risks.groupby("employee_id", group_keys=False)
            .apply(_within_emp)
            .sort_values(["employee_id", "date"])
        )

        if return_names_only:
            return (
                flight_risks[flight_risks["flight_risk"]][["employee_id"]]
                .drop_duplicates()
                .reset_index(drop=True)
            )
        else:
            return (
                flight_risks[flight_risks["flight_risk"]]
                .groupby("employee_id", as_index=False)
                .agg(
                    first_flight_risk=("date", "min"),
                    max_neg_sent_in_30d=("neg_30d", "max"),
                )
                .drop_duplicates()
                .reset_index(drop=True)[
                    ["employee_id", "first_flight_risk", "max_neg_sent_in_30d"]
                ]
            )

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
            self.df = df

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
        rankings = self.df.copy()
        rankings = rankings.sort_values(
            by=["date","sentiment_num","employee_id"],
            ascending=[True, False, True]
        ).groupby("date", as_index=False).head(3)
        return rankings

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
        rankings = self.df.copy()
        rankings = rankings.sort_values(
            by=["date","sentiment_num","employee_id"],
            ascending=[True, True, True]
        ).groupby("date", as_index=False).head(3)
        return rankings

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
