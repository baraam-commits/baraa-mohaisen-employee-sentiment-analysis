from transformers import pipeline
import torch

class SentimentLabeler:
    def __init__(self, df = None, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        
        # check for cuda
        print(torch.cuda.is_available())
        self.device = 0 if torch.cuda.is_available() else -1
        # load model and initialize task
        self.classifier = pipeline("sentiment-analysis", model=model_name, device=self.device)
        self.df = df

    def _label_sentiments(self, texts):
        results = self.classifier(texts)
        
        for classification in results:
            # any middle of the road scores are labeled as neutral (accounting for general overconfidence)
            if classification["score"] < 0.65 and classification["score"] > 0.55:
                classification['label'] = "NEUTRAL"
                
        # return the labels only as a list
        return [result['label'] for result in results]
    
    def get_sentiments(self):
        # return dataframe with additional sentiment column
        self.df["sentiment"] = self._label_sentiments(self.df["body"].to_list())
        return self.df
