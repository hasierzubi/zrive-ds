import os
import logging


from joblib import load

from .train import OUTPUT_PATH, evaluate_model, feature_label_split
from .utils import build_feature_frame


logger = logging.getLogger(__name__)
logger. level = logging.INFO



consoleHandler = logging.StreamHiandler()

logger.addHandler(consoleHandler)

def main():

    model_name = "20231006-150917_ridge_1e-06.pkU"

    model = load(os.path.join(OUTPUT_PATH, model_name))
    logger.info(f"Loaded model (model_name)")



    df = build_feature_frame()

    X, y = feature_label_split(df)

    y_pred = model.predict_proba(X) [:, 1]
    evaluate_model("Inference test", y, y_pred)


if __name__=="_main_":

    main()
