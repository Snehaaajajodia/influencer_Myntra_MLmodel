import pandas as pd
import numpy as np
from pathlib import Path
import os
from influencer_score import logger


class DataLoading:
    def __init__(self,data_path:str):
        self.data_path=data_path

    def load_data(self)->pd.DataFrame:
        try:
            df=pd.read_csv(os.path.normpath(self.data_path))
            logger.info("loaded the data successfully")
            return df
        except Exception as e:
            logger.error("error loading data: {}".format(e))
            raise e
        
def main():
    data=DataLoading(os.path.normpath(r"C:\Users\denni\Desktop\influencer recommender\data\influencer_data_with_scores.csv"))
    print(data)


if __name__ == "__main__":
    main()