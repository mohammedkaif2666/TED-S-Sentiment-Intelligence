import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load predictions
try:
    df = pd.read_excel(
        os.path.join(BASE_DIR, "experiments/output/predictions/brexitvote_08.00-13.59_predictions.xlsx")
    )
except FileNotFoundError:
    print("Prediction file not found. Please run the models first.")
    exit(1)

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

# Create grouped sentiment counts
sentiment_counts = (
    df.groupby(["timestamp", "predictions"])
    .size()
    .unstack(fill_value=0)
)

# Rolling average (trend smoothing)
rolling = sentiment_counts.rolling(window=20).mean()

# --------------------------
# 1. Sentiment Distribution
# --------------------------
plt.figure(figsize=(8, 6))
sns.countplot(
    data=df,
    x="predictions",
    palette="Set2"
)
plt.title("Sentiment Distribution")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "distribution.png"))
plt.close()

# --------------------------
# 2. Sentiment Timeline
# --------------------------
plt.figure(figsize=(8, 6))
for sentiment in sentiment_counts.columns:
    plt.plot(
        sentiment_counts.index,
        sentiment_counts[sentiment],
        label=sentiment
    )
plt.title("Sentiment Timeline")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "timeline.png"))
plt.close()

# --------------------------
# 3. Heatmap
# --------------------------
plt.figure(figsize=(8, 6))
heatmap_data = sentiment_counts.T
sns.heatmap(
    heatmap_data,
    cmap="coolwarm"
)
plt.title("Sentiment Intensity Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "heatmap.png"))
plt.close()

# --------------------------
# 4. Rolling Trend
# --------------------------
plt.figure(figsize=(8, 6))
for sentiment in rolling.columns:
    plt.plot(
        rolling.index,
        rolling[sentiment],
        label=sentiment
    )
plt.title("Smoothed Sentiment Trend")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "trend.png"))
plt.close()

print("Plots successfully saved to static/plots/")
