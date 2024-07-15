from influencer_score import logger
from influencer_score.pipeline.data_loading import  DataLoading
from influencer_score.pipeline.data_preprocessing import Data_Preprocessing
from influencer_score.pipeline.model_training import Model_trainer
from influencer_score.pipeline.prediction import Predictor
import pandas as pd
import xgboost as xgb
from typing import Tuple
import numpy as np
import json
STAGE_01="Data loading"

def data_loader()->pd.DataFrame:
    loader=DataLoading(r"data\influencer_data_with_scores.csv")
    df=loader.load_data()
    logger.info("Stage {} finished".format(STAGE_01))
    return df

STAGE_02="Data Preprocessing"

def data_preprocess(df:pd.DataFrame)->pd.DataFrame:
    preprocessing=Data_Preprocessing()
    df=preprocessing.preprocess(df)
    logger.info("Stage {} finished".format(STAGE_02))
    return df

STAGE_03="model training"

def model_trainer(df:pd.DataFrame)->Tuple[
    xgb.XGBRegressor,float,float
]:
    train=Model_trainer()
    model,mse,r2=train.trainer(df,xgb)
    logger.info("Stage {} finished".format(STAGE_03))
    return model,mse,r2

STAGE_04="Prediction"


# def Prediction(df:pd.DataFrame):
#     pred=Predictor()
#     y_pred=pred.prediction(df)
#     logger.info("Stage {} finished".format(STAGE_04))
#     return y_pred

# To test the flask app
def Prediction(df:dict):
    pred=Predictor()
    df = pd.DataFrame([df])
    y_pred=pred.prediction(df)
    logger.info("Stage {} finished".format(STAGE_04))
    return y_pred.tolist()

def main():
    df=data_loader()
    df=data_preprocess(df)
    model,mse,r2=model_trainer(df)
    scores_dict = {
    "Mean Squared Error (MSE)": mse,
    "R^2 Score": r2
    }
    json_file = "scorer.json"
    with open(json_file, 'w') as f:
        json.dump(scores_dict, f, indent=4)
    y_pred=Prediction(df)

if __name__=="__main__":
    main()