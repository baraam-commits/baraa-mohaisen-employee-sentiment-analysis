import pandas as pd 
import csv

class LoadData:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load_pandas_dataframe(self,clean=True):
        data_frame = pd.read_csv(self.file_path)
        
        # helper function to orginze and clean the dataframe.
        def _clean_df(df):
            # Combine Text: get rid of unnecessary spaces
            df["text"] = (df["Subject"].fillna("") + " " + df["body"].fillna("")).str.strip()
            
            # Convert date to datetime and sort by order for downstream processing
            df["dt"] = pd.to_datetime(df["date"], errors="coerce", infer_datetime_format=True)
            df = df.dropna(subset=["dt"])
            df = df.sort_values(["from", "dt"], kind="mergesort").reset_index(drop=True)
            
            # Normalize employee IDs by extracting from email addresses
            df["employee_id"] = df["from"].str.extract(r"([^@]+)").iloc[:,0].str.lower()

            # drop any duplicates
            df = df.drop_duplicates(subset=["employee_id", "dt", "text"])

            # Strip any newlines, multiple spaces, and non printables
            import re
            df["text"] = df["text"].apply(lambda s: re.sub(r"\s+", " ", s.strip()) if isinstance(s, str) else "")
            return df

        return _clean_df(data_frame) if clean else data_frame
