import pandas as pd 
import csv

class LoadData:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load_pandas_dataframe(self,clean=True):
        data_frame = pd.read_csv(self.file_path)
        
        # helper function to orginze and clean the dataframe.
        def _prep_df(df):
            
            

            
            if clean:

                # Combine Text: get rid of unnecessary spaces
                df["text"] = (df["Subject"].fillna("") + " " + df["body"].fillna("")).str.strip()
                
                # Normalize employee IDs by extracting from email addresses
                df["employee_id"] = df["from"].str.extract(r"([^@]+)").iloc[:,0].str.lower()
                
                # drop any duplicates
                df = df.drop_duplicates(subset=["employee_id", "date", "text"])

                # Strip any newlines, multiple spaces, and non printables
                import re
                df["text"] = df["text"].apply(lambda s: re.sub(r"\s+", " ", s.strip()) if isinstance(s, str) else "")
                
            else:
                df = df.dropna(subset=["date"])

            # Convert date to datetime and sort by order for downstream processing
            
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values(["from", "date"], kind="mergesort").reset_index(drop=True)
            
            return df

        return _prep_df(data_frame)
