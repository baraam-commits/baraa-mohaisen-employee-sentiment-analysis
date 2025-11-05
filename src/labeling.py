from transformers import pipeline
import pandas as pd
import torch
from tqdm import tqdm
from typing import List

class SentimentLabeler:
    """
    Task 2: 3-class sentiment labeling for employee messages.

    Runs a two-model pipeline to assign one of {NEGATIVE, NEUTRAL, POSITIVE}
    to each input text. A "trusted" 2-class model (SST-2) and a 3-class
    fallback model (RoBERTa) are combined via a weighted ensemble.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table. Must contain a 'text' column of strings.
    trusted_model_name : str, default "distilbert-base-uncased-finetuned-sst-2-english"
        Primary SST-2 model. Returns NEGATIVE or POSITIVE with probabilities.
    fallback_model_name : str, default "j-hartmann/sentiment-roberta-large-english-3-classes"
        Secondary model providing NEGATIVE / NEUTRAL / POSITIVE with probabilities.

    Attributes
    ----------
    df : pandas.DataFrame
        Working DataFrame. A new 'sentiment' column is appended by `get_sentiments`.
    device : int
        0 if CUDA is available, else −1 for CPU.
    trusted_classifier : transformers.pipelines.TextClassificationPipeline
        Primary sentiment pipeline with `return_all_scores=True`.
    fallback_classifier : transformers.pipelines.TextClassificationPipeline
        Secondary sentiment pipeline with `return_all_scores=True`.
    label_map : dict
        Normalizes model-specific labels to {"NEGATIVE","NEUTRAL","POSITIVE"}.

    Notes
    -----
    - The trusted model is 2-class. Neutral is supplied by the fallback model
      and chosen by the ensemble when its confidence outweighs the trusted top-1.
    - GPU is used if `torch.cuda.is_available()` is True.
    - This class mutates only its internal copy (self.df). Callers get a new
      column returned from `get_sentiments()`.
    - For reproducibility across runs, set Python, PyTorch, and Transformers
      seeds at the application entry point if needed.

    Examples
    --------
    >>> labeler = SentimentLabeler(df)  # df has a 'text' column
    >>> out = labeler.get_sentiments()
    >>> out[['text','sentiment']].head()
    """

    def __init__(self, df: pd.DataFrame,
                 trusted_model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
                 fallback_model_name: str = "j-hartmann/sentiment-roberta-large-english-3-classes"):
        """
        Initialize pipelines and normalize device.

        Parameters
        ----------
        df : pandas.DataFrame
            Must include a 'text' column.
        trusted_model_name : str
            HF model id for the 2-class SST-2 pipeline.
        fallback_model_name : str
            HF model id for the 3-class RoBERTa pipeline.

        Side Effects
        ------------
        - Prints CUDA availability for quick diagnostics.

        Raises
        ------
        OSError
            If model weights cannot be loaded by Transformers.
        """
        self.df = df

        use_cuda = torch.cuda.is_available()
        self.device = 0 if use_cuda else -1
        print(f"CUDA available: {use_cuda}")

        self.trusted_classifier = pipeline(
            "sentiment-analysis",
            model=trusted_model_name,
            device=self.device,
            return_all_scores=True
        )
        self.fallback_classifier = pipeline(
            "text-classification",
            model=fallback_model_name,
            device=self.device,
            return_all_scores=True
        )

        # Normalize model label formats
        self.label_map = {
            "LABEL_0": "NEGATIVE", "LABEL_1": "NEUTRAL", "LABEL_2": "POSITIVE",
            "negative": "NEGATIVE", "neutral": "NEUTRAL", "positive": "POSITIVE",
            "NEGATIVE": "NEGATIVE", "NEUTRAL": "NEUTRAL", "POSITIVE": "POSITIVE",
            "0": "NEGATIVE", "1": "POSITIVE"
        }

    def _label_sentiments(self, texts: list[str]) -> list[str]:
        """
        Batch inference with both models. Returns normalized labels.

        Parameters
        ----------
        texts : list[str]
            The input texts to classify.

        Returns
        -------
        list[str]
            One label per input in {"NEGATIVE","NEUTRAL","POSITIVE"}.

        Process
        -------
        1) Run `trusted_classifier` and `fallback_classifier` with return_all_scores.
        2) For each item i:
           - Map model labels to normalized set via `self.label_map`.
           - Sort per-model scores desc.
           - Call `_weighted_two_model_ensemble_classification(trusted_scores, fallback_scores)`.
        3) Collect final labels.

        Notes
        -----
        - Assumes both pipelines return lists of equal length. If upstream
          batching diverges, reconcile lengths before looping.
        - Progress bars can be added around batching with `tqdm` at call sites.

        Examples
        --------
        >>> SentimentLabeler(df)._label_sentiments(["ok","bad","great"])
        ['NEUTRAL','NEGATIVE','POSITIVE']
        """
        trusted_results = self.trusted_classifier(texts)
        fallback_results = self.fallback_classifier(texts)
        return_results = []

        for i in range(len(trusted_results) or len(fallback_results)):
            trusted_scores = [
                [self.label_map.get(entry["label"], entry["label"]).upper(), entry["score"]]
                for entry in trusted_results[i]
            ]
            trusted_scores.sort(key=lambda x: x[1], reverse=True)

            fallback_scores = [
                [self.label_map.get(entry["label"], entry["label"]).upper(), entry["score"]]
                for entry in fallback_results[i]
            ]
            fallback_scores.sort(key=lambda x: x[1], reverse=True)

            return_results.append(
                self._weighted_two_model_ensemble_classification(trusted_scores, fallback_scores)
            )

            # Optional: tqdm update here if wrapping loop with a progress bar

        return return_results

    def _weighted_two_model_ensemble_classification(
        self,
        cl1: List[list],
        cl2: List[list],
        w1: float = 1.3,
        w2: float = 1.0
    ) -> str:
        """
        Weighted two-model ensemble for final label selection.

        Parameters
        ----------
        cl1 : list[list]
            Trusted model top-k, as [[label, score], ...], sorted desc.
            Expected length 2 for SST-2.
        cl2 : list[list]
            Fallback model top-k, as [[label, score], ...], sorted desc.
            Expected length 3 for 3-class model.
        w1 : float, default 1.3
            Weight on trusted model top-1 confidence.
        w2 : float, default 1.0
            Weight on trusted top-1 in the secondary check.

        Returns
        -------
        str
            Final label in {"NEGATIVE","NEUTRAL","POSITIVE"}.

        Decision Rule
        -------------
        1) Start with trusted top-1.
        2) If (trusted_top1 * w1) < (fallback_top1 + fallback_third), select fallback_top1.
        3) Else if fallback second label equals trusted second label AND
           (fallback_top1 + fallback_third) > (w2 * trusted_top1), select fallback_top1.
        4) Else keep trusted top-1.

        Notes
        -----
        - This rule promotes NEUTRAL when the 3-class model is confident and the
          trusted model is comparatively weak.
        - Assumes input lists are already score-sorted descending and contain the
          expected number of entries.

        Examples
        --------
        >>> cl1 = [["POSITIVE", 0.55], ["NEGATIVE", 0.45]]
        >>> cl2 = [["NEUTRAL", 0.64], ["POSITIVE", 0.22], ["NEGATIVE", 0.14]]
        >>> SentimentLabeler(df=None)._weighted_two_model_ensemble_classification(cl1, cl2)
        'NEUTRAL'
        """
        weighted_classification = cl1[0][0]

        if (cl1[0][1] * w1) < (cl2[0][1] + cl2[2][1]):
            weighted_classification = cl2[0][0]
        elif (cl2[1][0] == cl1[1][0]) and ((cl2[0][1] + cl2[2][1]) > w2 * cl1[0][1]):
            weighted_classification = cl2[0][0]

        return weighted_classification

    def get_sentiments(self) -> pd.DataFrame:
        """
        Append a 'sentiment' column to `self.df` using the two-model ensemble.

        Returns
        -------
        pandas.DataFrame
            Copy of `self.df` with a new 'sentiment' column.

        Raises
        ------
        ValueError
            If the 'text' column is missing.

        Notes
        -----
        - The method converts 'text' to `str` before inference.
        - For large datasets, consider chunking `self.df['text']` and concatenating
          results to control memory and get progress reporting.

        Examples
        --------
        >>> labeled = SentimentLabeler(df).get_sentiments()
        >>> labeled['sentiment'].value_counts()
        """
        if "text" not in self.df.columns:
            raise ValueError("DataFrame must contain a 'text' column.")
        self.df["sentiment"] = self._label_sentiments(self.df["text"].astype(str).tolist())
        return self.df
