import pandas as pd
import numpy as np
from pathlib import Path
import os
import logging
import xgboost as xgb 
from influencer_score.pipeline.data_preprocessing import Data_Preprocessing
class Predictor:
    def __init__(self):
        pass
    def prediction(self,df:pd.DataFrame):

        # feature_order = ['Number of subscribers', 'followers', 'average_likes', 'average_comments', 'average_shares', 'budget_per_video', 'engagement_rate', 'follower_growth_rate', 'primary_platform_Facebook', 'primary_platform_Instagram', 'primary_platform_YouTube', 'Targetaudience_gender_Female', 'Targetaudience_gender_Male', 'Targetaudience_gender_Others', 'product_category_Accessories', 'product_category_Clothing', 'product_category_Cosmetics', 'Geographical location: state of India_Gujarat', 'Geographical location: state of India_Kerala', 'Geographical location: state of India_Maharashtra', 'Geographical location: state of India_Punjab', 'Geographical location: state of India_Rajasthan', 'Geographical location: state of India_Sikkim', 'Geographical location: state of India_Tamil Nadu', 'LL_age', 'UL_age', 'campaign duration_transform']
        
        # df = df[feature_order]
        # df=pd.read_csv(os.path.normpath(r'data/influencer_data_with_scores.csv'))
        # process=Data_Preprocessing()
        # df=process.preprocess(df)
        model = xgb.XGBRegressor()
        model.load_model(os.path.normpath(r'model\model.xgb'))
        try:
            y_pred=model.predict(df.drop(columns=['influencer_id','score']))
        except:
            y_pred=model.predict(df)
        sorted_indices = np.argsort(y_pred)[::-1]
        sorted_y_pred = y_pred[sorted_indices]

        indices_array = np.zeros_like(sorted_indices)

        # Store indices in the array
        for idx, value in enumerate(sorted_y_pred):
            indices_array[idx] = sorted_indices[idx]
        return indices_array
    


        