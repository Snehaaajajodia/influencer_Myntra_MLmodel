import pandas as pd
import numpy as np
from pathlib import Path
import os
import logging
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from typing import Tuple

class Model_trainer:
    def __init__(self):
        pass

    def trainer(self,df:pd.DataFrame,model:xgb.XGBRegressor)->Tuple[
        xgb.XGBRegressor,float,float
    ]:
        X = df.drop(columns=['influencer_id', 'score'])
        y = df['score']

        # Splitting data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Creating the XGBoost model
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, seed=42)

        # Training the model
        model.fit(X_train, y_train)

        # Predicting on the test set
        y_pred = model.predict(X_test)

        # Evaluating the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        output_dir=os.path.normpath(r'C:\Users\denni\Desktop\influencer recommender\model')
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "model.xgb")
        model.save_model(model_path)
        return model,mse,r2
    
