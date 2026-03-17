
# Modified Transformer Experiment for Aggregated Sentiment Dataset

import logging
import os
import pandas as pd

from algo.util.file_util import create_folder_if_not_exist
from experiments import transformer_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIRECTORY = os.path.join(BASE_PATH, "output")
PREDICTION_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "predictions")


def predict(data_file_path, predictions_folder):

    create_folder_if_not_exist(predictions_folder, is_file_path=False)

    file_name = os.path.splitext(os.path.basename(data_file_path))[0]

    logger.info(f"Loading dataset: {data_file_path}")

    # Read Excel dataset
    data = pd.read_excel(data_file_path)

    logger.info(f"Dataset shape: {data.shape}")

    # Determine sentiment using aggregated scores
    data["predictions"] = data[
        ["negative_mean", "neutral_mean", "positive_mean"]
    ].idxmax(axis=1)

    # Remove "_mean"
    data["predictions"] = data["predictions"].str.replace("_mean", "")

    # Convert id column to string
    if "id" in data.columns:
        data["id"] = data["id"].astype(str)

    output_file = os.path.join(
        predictions_folder,
        f"{file_name}_transformer_predictions.xlsx"
    )

    data.to_excel(output_file, sheet_name="Sheet1", index=False)

    logger.info(f"Predictions saved to {output_file}")


if __name__ == "__main__":

    # Dataset paths
    munliv_file = os.path.join(
        os.path.dirname(BASE_PATH),
        "data",
        "semi-supervised",
        "munliv_15.28-17.23.xlsx"
    )

    brexitvote_file = os.path.join(
        os.path.dirname(BASE_PATH),
        "data",
        "semi-supervised",
        "brexitvote_08.00-13.59.xlsx"
    )

    predictions_folder = PREDICTION_DIRECTORY

    predict(munliv_file, predictions_folder)
    predict(brexitvote_file, predictions_folder)