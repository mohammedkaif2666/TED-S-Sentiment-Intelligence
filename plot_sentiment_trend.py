import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# Load predictions
df = pd.read_excel(
    "experiments/output/predictions/brexitvote_08.00-13.59_predictions.xlsx"
)

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

# Create professional dashboard
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# --------------------------
# 1. Sentiment Distribution
# --------------------------
sns.countplot(
    data=df,
    x="predictions",
    palette="Set2",
    ax=axes[0,0]
)

axes[0,0].set_title("Sentiment Distribution")

# --------------------------
# 2. Sentiment Timeline
# --------------------------
for sentiment in sentiment_counts.columns:
    axes[0,1].plot(
        sentiment_counts.index,
        sentiment_counts[sentiment],
        label=sentiment
    )

axes[0,1].set_title("Sentiment Timeline")
axes[0,1].legend()

# --------------------------
# 3. Heatmap
# --------------------------
heatmap_data = sentiment_counts.T

sns.heatmap(
    heatmap_data,
    cmap="coolwarm",
    ax=axes[1,0]
)

axes[1,0].set_title("Sentiment Intensity Heatmap")

# --------------------------
# 4. Rolling Trend
# --------------------------
for sentiment in rolling.columns:
    axes[1,1].plot(
        rolling.index,
        rolling[sentiment],
        label=sentiment
    )

axes[1,1].set_title("Smoothed Sentiment Trend")
axes[1,1].legend()

plt.tight_layout()
plt.show()