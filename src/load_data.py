import pandas as pd 
import csv

class LoadData:
    """Takes in a CSV file path with body, from, date, Subject columns and returns a pandas dataframe.
    """
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load_pandas_dataframe(self,clean=True):
        """returns a pandas dataframe of CSV normalized depending on clean bool, adds text, employee_id columns.
            Sorts by date per employee entry. 

        Args:
            clean (bool, optional): Whether to clean data in columns set to false to reduce processing time IF data is already clean. Defaults to True.

        Returns:
            pd.DataFrame: Normalized pandas DataFrame. with text, employee_id columns added and sorted by date.
        """
        data_frame = pd.read_csv(self.file_path)
        
        # helper function to organize and clean the dataframe.
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
