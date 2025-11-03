from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

class SentimentLabeler:
    """
    SentimentLabeler:
    Uses a 3-class sentiment model (POSITIVE / NEUTRAL / NEGATIVE).
    """

    def __init__(self, df=None,
                trusted_model_name="distilbert-base-uncased-finetuned-sst-2-english",
                fallback_model_name="j-hartmann/sentiment-roberta-large-english-3-classes"):
        self.df = df
        

        # Detect GPU
        use_cuda = torch.cuda.is_available()
        self.device = 0 if use_cuda else -1
        print(f"CUDA available: {use_cuda}")


        self.trusted_classifier = pipeline("sentiment-analysis", model=trusted_model_name, device= self.device, return_all_scores=True)
        self.fallback_classifier = pipeline("text-classification", model=fallback_model_name, device= self.device,return_all_scores=True)

        # Normalize model label formats
        self.label_map = {
            "LABEL_0": "NEGATIVE", "LABEL_1": "NEUTRAL", "LABEL_2": "POSITIVE",
            "negative": "NEGATIVE", "neutral": "NEUTRAL", "positive": "POSITIVE",
            "NEGATIVE": "NEGATIVE", "NEUTRAL": "NEUTRAL", "POSITIVE": "POSITIVE",
            "0": "NEGATIVE", "1": "POSITIVE"
        }

    def _label_sentiments(self, texts: list[str]) -> list[str]:
        """Run batched inference with progress bar and return normalized labels."""
        trusted_results = self.trusted_classifier(texts)
        fallback_results = self.fallback_classifier(texts)
        return_results = []
        
        for i in range(len(trusted_results) or len(fallback_results)):
            # trusted model results for this text
            trusted_scores = [
                [self.label_map.get(entry["label"], entry["label"]).upper(), entry["score"]]
                for entry in trusted_results[i]
            ]
            trusted_scores.sort(key=lambda x: x[1], reverse=True)

            # fallback model results for this text
            fallback_scores = [
                [self.label_map.get(entry["label"], entry["label"]).upper(), entry["score"]]
                for entry in fallback_results[i]
            ]
            fallback_scores.sort(key=lambda x: x[1], reverse=True)
            return_results.append(self._weighted_two_model_ensemble_classification(trusted_scores,fallback_scores))
            
        return return_results

    def _weighted_two_model_ensemble_classification(self,cl1:list,cl2:list, w1 = 1.3, w2 = 1.0) -> str:

        weighted_classification = cl1[0][0]
        
        if(cl1[0][1] * w1 < (cl2[0][1] + cl2[2][1])):
            weighted_classification = cl2[0][0]
        elif (cl2[1][0] == cl1[1][0]) and ((cl2[0][1] + cl2[2][1]) > w2*cl1[0][1]):
            weighted_classification = cl2[0][0]
        
        return weighted_classification
    
    
    def get_sentiments(self):
        """Append sentiment column to dataframe."""
        if "text" not in self.df.columns:
            raise ValueError("DataFrame must contain a 'text' column.")
        self.df["sentiment"] = self._label_sentiments(self.df["text"].astype(str).tolist())
        return self.df
