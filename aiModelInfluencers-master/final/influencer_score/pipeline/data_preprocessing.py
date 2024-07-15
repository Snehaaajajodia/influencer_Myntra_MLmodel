import pandas as pd
import numpy as np
from pathlib import Path
import os
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from influencer_score.pipeline.data_loading import DataLoading

class Data_Preprocessing:
    def __init__(self):
        pass

    @staticmethod
    def year_to_month(text):
        if("year" in text):
            return int(text.split(" ")[0])*12
        else:
            return int(text.split(" ")[0])
        
    def preprocess(self,df:pd.DataFrame)->pd.DataFrame:
        cols_to_drop=['past_campaign_success','pat_campaign_success_rate']
        df.drop(columns=cols_to_drop,inplace=True)
        categorical_cols = ['primary_platform', 'Targetaudience_gender', 'product_category','Geographical location: state of India']
        df_encoded = pd.get_dummies(df[categorical_cols],dtype=np.int16)
        df=pd.concat([df.drop(columns=categorical_cols),df_encoded],axis=1)
        df['LL_age']=df['Targeted_audience_age'].apply(lambda x:x.split("to")[0]).astype( np.int16)
        df['UL_age']=df['Targeted_audience_age'].apply(lambda x:x.split("to")[1]).astype(np.int16)
        df.drop(columns=['Targeted_audience_age'],inplace=True)
        df['campaign duration_transform']=df['campaign duration'].apply(self.year_to_month)
        df.drop(columns='campaign duration',inplace=True)
        columns_to_normalize = [
            'Number of subscribers', 'followers', 'average_likes', 'average_comments',
            'average_shares', 'budget_per_video', 'engagement_rate',
            'follower_growth_rate', 'LL_age', 'UL_age', 'campaign duration_transform'
        ]
        scaler = MinMaxScaler()
        df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
        return df
    

def main():
    df=DataLoading(os.path.normpath(r"C:\Users\denni\Desktop\influencer recommender\data\influencer_data_with_scores.csv"))
    preprocessing=Data_Preprocessing()
    df=preprocessing.preprocess(df)
    

if __name__ == "__main__":
    main()