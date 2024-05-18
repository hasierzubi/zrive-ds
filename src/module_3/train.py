import pandas as pd
import os
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, auc

from sklearn.preprocessing import StandardScaler
import joblib
import datetime
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator

from .utils import build_feature_frame, STORAGE_PATH

logger = logging.getLogger(__name__)
logger.level = logging.INFO

consolo_handler = logging.StreamHandler()
logger.addHandler(consolo_handler)

HOLDOUT_SIZE = 0.2
RIDGE_Cs = [1e-8, 1e-6, 1e-4, 1e-2]
FEATURE_COLS = [
    "ordered_before",
    "abandoned_before",
    "global_popilarity",
    "set_as_regular",
]

LABEL_COL = "outcome"

OUTPUT_PATH = os.path.join(STORAGE_PATH, "module_3_models")

def evaluate_model( model_name: str, y_test: pd.Series, y_pred: pd.Series) -> float:
    """
    evaluate model based on precision recall AUC. We use ROC AUC as a secondary metric

    """

    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_test, y_pred)
    logger.info(
        f"{model_name} results - PR AUC: {pr_auc:.2f}, ROC AUC: {roc_auc:.2f}"
    )
    return pr_auc

    






def feature_label_split(df: pd.DataFrame) -> (pd.DataFrame, pd.Series): #Tuple[pd.DataFrame, pd.Series]
    return df[FEATURE_COLS], df[LABEL_COL]

def train_test_split(
    df: pd.DataFrame,train_size: float
) -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    """
    there is a time component in the data, so we will split the data based on time
    """
    daily_orders = df.groupby("order_date").order_id.nunique()
    cumsum_daily_orders = daily_orders.cumsum() / daily_orders.sum() #cumsum es la suma acumulada
    cutoff = cumsum_daily_orders[cumsum_daily_orders <= train_size].idxmax()

    X_train, y_train = feature_label_split(df[df.order_date <= cutoff])
    X_val, y_val = feature_label_split(df[df.order_date > cutoff])
    logger.info(
        "splitting data on {}, {} train samples, {} validation samples".format(
            cutoff, X_train.shape[0], X_val.shape[0]
        )
    )
    return X_train, y_train, X_val, y_val

def save_model(model: BaseEstimator, model_name: str) -> None:
    logger.info(f"Saving model {model_name} to {OUTPUT_PATH}")
    if not os.path.exists(OUTPUT_PATH):
        logger.info(f"Creating directory {OUTPUT_PATH}")
        os.makedirs(OUTPUT_PATH)

    model_path = f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{model_name}.pkl"
    joblib.dump(model, os.path.join(OUTPUT_PATH, model_name))

def ridge_model_selection(df: pd.DataFrame) -> None:
    """
    after exploration we foun that some strong regularisation seemed to improve the model. However, we prefere to do some selection here with every retrain to make sure
    thats still the optimal hyperparameter
    """
    train_size = 1 - HOLDOUT_SIZE
    X_train, y_train, X_val, y_val = train_test_split(df, train_size=train_size)

    best_auc = 0
    for c in RIDGE_Cs:
        logger.info(f"Training Ridge model with C={C}")
        lr = make_pipeline(StandardScaler(), LogisticRegression(penalty="l2", C=c))
        lr.fit(X_train, y_train)
        _ = evaluate_model(
            f"Ridge model with C={c}", y_test=y_train, y_pred=lr.predict_proba(X_train)[:, 1]
        )
        pr_auc = evaluate_model(
            f"Ridge model with C={c}", y_test=y_val, y_pred=lr.predict_proba(X_val)[:, 1]
        )
        if pr_auc > best_auc:
            logger.info(f"New best model found with C={c}")
            best_auc = pr_auc
            best_c = c
    logger.info(f"Training final model with C={best_c} over whole dataset")
    best_model = make_pipeline(
        StandardScaler(), LogisticRegression(penalty="l2", C=best_c)
    )
    X, y = feature_label_split(df)
    best_model.fit(X, y) 

    save_model(best_model, f"ridge_{best_c}")       
    

def main():
    feature_frame = build_feature_frame()
    ridge_model_selection(feature_frame)

if __name__ == "__main__":
    main()
