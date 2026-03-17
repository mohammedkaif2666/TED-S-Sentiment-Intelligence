
# Modified TextBlob Experiment for Aggregated Sentiment Dataset

import logging
import os
import pandas as pd

from algo.util.file_util import create_folder_if_not_exist

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIRECTORY = os.path.join(BASE_PATH, "output")
PREDICTION_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "predictions")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_sentiment(row):
    scores = {
        "negative": row["negative_mean"],
        "neutral": row["neutral_mean"],
        "positive": row["positive_mean"],
    }
    return max(scores, key=scores.get)


def predict(data_file_path, predictions_folder):

    create_folder_if_not_exist(predictions_folder, is_file_path=False)

    file_name = os.path.splitext(os.path.basename(data_file_path))[0]

    # Read Excel dataset
    data = pd.read_excel(data_file_path)

    # Generate predictions
    data["predictions"] = data.apply(get_sentiment, axis=1)

    # Ensure id is string
    if "id" in data.columns:
        data["id"] = data["id"].astype(str)

    # Save output
    output_file = os.path.join(predictions_folder, f"{file_name}_textblob_predictions.xlsx")

    data.to_excel(output_file, sheet_name="Sheet1", index=False)

    logger.info(f"Predictions saved to {output_file}")


if __name__ == "__main__":

    brexitvote_test_file = os.path.join(
        os.path.dirname(BASE_PATH),
        "data",
        "semi-supervised",
        "brexitvote_08.00-13.59.xlsx",
    )

    predictions_folder = PREDICTION_DIRECTORY

    predict(brexitvote_test_file, predictions_folder)
