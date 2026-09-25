"""
Task 3 - Natural language processing: sentiment analysis of social media posts.

Steps: clean the text, tokenize, remove stopwords, lemmatize, score every post with two
tools (TextBlob and NLTK's VADER), check both against the emotions given in the file, keep
the better one, then look at the distribution and at the words with word clouds.

Run it from the project folder, after 01_clean_data.py:
    python src/03_sentiment_analysis.py

Needs internet the first time (nltk downloads its word lists).
"""
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, classification_report
from textblob import TextBlob
from wordcloud import WordCloud

for package in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4", "vader_lexicon"):
    nltk.download(package, quiet=True)

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")
COLORS = {"Positive": "seagreen", "Neutral": "gray", "Negative": "crimson"}
ORDER = ["Positive", "Neutral", "Negative"]

df = pd.read_csv(ROOT / "data" / "sentiment_clean.csv", parse_dates=["timestamp"])
print(f"{len(df)} posts, {df['platform'].nunique()} platforms, {df['country'].nunique()} countries")


# 1. text preprocessing -------------------------------------------------------------
def light_clean(text):
    """lower case, no links, mentions, hashtag signs or digits. Stopwords are kept on purpose
    because a word like 'not' changes the sentiment."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = text.replace("#", " ")
    text = re.sub(r"[^a-z'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def to_tokens(clean_text):
    """tokenization -> stopwords removal -> lemmatization (used for the word analysis)"""
    tokens = word_tokenize(clean_text)
    tokens = [t for t in tokens if t.isalpha() and len(t) > 2 and t not in stop_words]
    # verb first, then noun, so that 'celebrating' -> 'celebrate' and 'friends' -> 'friend'
    return [lemmatizer.lemmatize(lemmatizer.lemmatize(t, pos="v"), pos="n") for t in tokens]


df["clean_text"] = df["text"].apply(light_clean)
df["tokens"] = df["clean_text"].apply(to_tokens)

example = df.iloc[3]
print("\nExample of preprocessing:")
print("  original:", example["text"])
print("  cleaned :", example["clean_text"])
print("  tokens  :", example["tokens"])


# 2. two sentiment scorers -----------------------------------------------------------
# TextBlob: polarity from -1 to +1, computed on the cleaned text.
# VADER (in nltk): compound score from -1 to +1, made for short social media posts. It uses
# capital letters and punctuation, so it gets the original text.
# Both use the usual +-0.05 neutral band around zero.
def to_label(score):
    if score >= 0.05:
        return "Positive"
    if score <= -0.05:
        return "Negative"
    return "Neutral"


vader = SentimentIntensityAnalyzer()
df["textblob_score"] = df["clean_text"].apply(lambda t: TextBlob(t).sentiment.polarity)
df["vader_score"] = df["text"].apply(lambda t: vader.polarity_scores(t)["compound"])
df["textblob_sentiment"] = df["textblob_score"].apply(to_label)
df["vader_sentiment"] = df["vader_score"].apply(to_label)

print("\nDistribution of the labels:")
print(pd.DataFrame({
    "TextBlob": df["textblob_sentiment"].value_counts().reindex(ORDER),
    "VADER": df["vader_sentiment"].value_counts().reindex(ORDER),
}).to_string())
print(f"the two tools agree on {(df['textblob_sentiment'] == df['vader_sentiment']).mean():.1%} of the posts")


# 3. which one is better? Check against the emotions of the file ----------------------
# The file has no positive/negative/neutral column, only 190+ emotions (Joy, Grief, ...).
# To have something to compare with, I grouped them by hand. It is my own judgment and some
# emotions are ambiguous (nostalgia, surprise...), so those went to Neutral. The full list is
# saved in results/emotion_mapping.csv if you disagree with a choice.
POSITIVE = """Positive Joy Excitement Contentment Gratitude Serenity Happy Hopeful Awe Pride Elation
Euphoria Enthusiasm Determination Playful Inspiration Happiness Hope Empowerment Inspired Admiration
Calmness Compassion Tenderness Fulfillment Reverence Proud Grateful Compassionate Thrill Enchantment
Love Amusement Kind Empathetic Free-spirited Confident Satisfaction Accomplishment Harmony Creativity
Wonder Adventure Enjoyment Affection Adoration Zest Radiance Rejuvenation Coziness Resilience
Tranquility Overjoyed Motivation JoyfulReunion Blessed Appreciation Confidence Wonderment Optimism
Mindfulness PlayfulJoy DreamChaser Elegance FestiveJoy Freedom Dazzle Adrenaline ArtisticBurst
CulinaryOdyssey Spark Marvel Positivity Kindness Friendship Success Amazement Romance Grandeur Energy
Celebration Charm Ecstasy Colorful Connection Iconic Engagement Touched Heartwarming Solace
Breakthrough Imagination Vibrancy Mesmerizing Triumph Captivation Melodic Hypnotic Whimsy Relief""".split()
POSITIVE += ["Joy in Baking", "Culinary Adventure", "Winter Magic", "Thrilling Journey", "Nature's Beauty",
             "Celestial Wonder", "Creative Inspiration", "Runway Creativity", "Ocean's Freedom"]

NEGATIVE = """Despair Grief Loneliness Sad Embarrassed Frustration Regret Numbness Melancholy Hate Bad
Disgust Bitterness Frustrated Betrayal Negative Boredom Overwhelmed Desolation Bitter Shame Jealousy
Resentment Fearful Jealous Devastated Envious Dismissive Heartbreak Anger Fear Sadness Disappointed
Anxiety Intimidation Helplessness Envy Apprehensive Isolation Disappointment Sorrow Loss Suffering
EmotionalStorm LostLove Exhaustion Darkness Desperation Ruins Heartache Obstacle Pressure
Miscalculation Confusion""".split()

# everything else (Neutral, Curiosity, Acceptance, Nostalgia, Surprise, Indifference...) -> Neutral
emotion_group = {e: "Positive" for e in POSITIVE}
emotion_group.update({e: "Negative" for e in NEGATIVE})
df["emotion_group"] = df["emotion"].map(emotion_group).fillna("Neutral")

mapping = (df.groupby(["emotion", "emotion_group"]).size().reset_index(name="posts")
           .sort_values(["emotion_group", "posts"], ascending=[True, False]))
mapping.to_csv(RESULTS_DIR / "emotion_mapping.csv", index=False)
print("\nEmotions grouped by hand:", df["emotion_group"].value_counts().reindex(ORDER).to_dict())

clear = df["emotion_group"] != "Neutral"     # the cases where the grouping is not debatable
for name in ("textblob", "vader"):
    col = f"{name}_sentiment"
    acc = accuracy_score(df["emotion_group"], df[col])
    acc_clear = accuracy_score(df.loc[clear, "emotion_group"], df.loc[clear, col])
    print(f"{name:>8}: accuracy {acc:.3f} on all posts | {acc_clear:.3f} on clearly positive/negative posts")

print("\nVADER, details per class:")
print(classification_report(df["emotion_group"], df["vader_sentiment"], labels=ORDER, digits=2, zero_division=0))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
for ax, name, title in zip(axes, ["textblob", "vader"], ["TextBlob", "VADER"]):
    cm = pd.crosstab(df["emotion_group"], df[f"{name}_sentiment"]).reindex(index=ORDER, columns=ORDER, fill_value=0)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    acc = accuracy_score(df["emotion_group"], df[f"{name}_sentiment"])
    ax.set_title(f"{title} (accuracy {acc:.0%})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Emotion group from the file")
fig.tight_layout()
fig.savefig(FIG_DIR / "nlp_1_textblob_vs_vader.png", dpi=130)
plt.close(fig)

# a few posts where TextBlob is wrong and VADER is right
wrong = df[(df["emotion_group"] == "Positive") & (df["textblob_sentiment"] == "Negative")
           & (df["vader_sentiment"] == "Positive")]
print(f"\nPositive posts that TextBlob calls negative and VADER gets right: {len(wrong)}. Some of them:")
for _, row in wrong.head(3).iterrows():
    print(f"  {row['text']}  (TextBlob {row['textblob_score']:.2f}, VADER {row['vader_score']:.2f})")

# from here on, VADER is the sentiment used in the analysis
df["sentiment"] = df["vader_sentiment"]


# 4. charts --------------------------------------------------------------------------
counts = df["sentiment"].value_counts().reindex(ORDER)
print("\nFinal sentiment distribution (VADER):")
print(pd.DataFrame({"posts": counts, "share_%": (counts / len(df) * 100).round(1)}).to_string())

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.countplot(data=df, x="sentiment", order=ORDER, hue="sentiment", palette=COLORS, legend=False, ax=axes[0])
axes[0].set_title("Sentiment of the posts (VADER)")
axes[0].set_xlabel("")
axes[0].set_ylabel("Number of posts")
for container in axes[0].containers:
    axes[0].bar_label(container)
sns.histplot(df["vader_score"], bins=30, color="steelblue", ax=axes[1])
axes[1].axvline(0.05, color="gray", linestyle="--")
axes[1].axvline(-0.05, color="gray", linestyle="--")
axes[1].set_title("VADER score of the posts")
axes[1].set_xlabel("Compound score (-1 very negative, +1 very positive)")
fig.tight_layout()
fig.savefig(FIG_DIR / "nlp_2_sentiment_distribution.png", dpi=130)
plt.close(fig)

by_platform = pd.crosstab(df["platform"], df["sentiment"], normalize="index")[ORDER] * 100
ax = by_platform.plot(kind="bar", stacked=True, color=[COLORS[c] for c in ORDER], figsize=(8, 5))
ax.set_title("Sentiment by platform")
ax.set_xlabel("")
ax.set_ylabel("Share of posts (%)")
plt.xticks(rotation=0)
ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3)
plt.tight_layout()
plt.savefig(FIG_DIR / "nlp_3_sentiment_by_platform.png", dpi=130)
plt.close()
print("\nSentiment by platform (%):")
print(by_platform.round(1).to_string())


def top_words(sentiment, n=15):
    words = Counter(w for tokens in df.loc[df["sentiment"] == sentiment, "tokens"] for w in tokens)
    return words.most_common(n)


fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
for ax, sentiment in zip(axes, ["Positive", "Negative"]):
    words, freq = zip(*top_words(sentiment))
    sns.barplot(x=list(freq), y=list(words), color=COLORS[sentiment], ax=ax)
    ax.set_title(f"Most frequent words - {sentiment} posts")
    ax.set_xlabel("Occurrences")
fig.tight_layout()
fig.savefig(FIG_DIR / "nlp_4_top_words.png", dpi=130)
plt.close(fig)
print("\nTop words, positive:", [w for w, _ in top_words("Positive", 10)])
print("Top words, negative:", [w for w, _ in top_words("Negative", 10)])


def word_cloud(tokens_series, color):
    text = " ".join(w for tokens in tokens_series for w in tokens)
    # one flat color per cloud, the pale shades of a colormap are hard to read
    return WordCloud(width=900, height=500, background_color="white", max_words=100,
                     color_func=lambda *args, **kwargs: color, random_state=42).generate(text)


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (title, series, color) in zip(axes, [
    ("All posts", df["tokens"], "steelblue"),
    ("Positive posts", df.loc[df["sentiment"] == "Positive", "tokens"], COLORS["Positive"]),
    ("Negative posts", df.loc[df["sentiment"] == "Negative", "tokens"], COLORS["Negative"]),
]):
    ax.imshow(word_cloud(series, color), interpolation="bilinear")
    ax.set_title(title)
    ax.axis("off")
fig.tight_layout()
fig.savefig(FIG_DIR / "nlp_5_word_clouds.png", dpi=130)
plt.close(fig)

out_cols = ["text", "emotion", "emotion_group", "platform", "country", "timestamp", "likes", "retweets",
            "textblob_score", "textblob_sentiment", "vader_score", "vader_sentiment", "sentiment"]
scored = df[out_cols].copy()
scored[["textblob_score", "vader_score"]] = scored[["textblob_score", "vader_score"]].round(4)
scored.to_csv(RESULTS_DIR / "sentiment_scored.csv", index=False)
print(f"\nFigures in {FIG_DIR}, tables in {RESULTS_DIR}")
