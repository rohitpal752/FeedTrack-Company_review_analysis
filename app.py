import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob

# Title
st.title("Indian Company Dataset")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Ambition Box.csv")
    return df

df = load_data()

# Filter only Swiggy reviews
swiggy_df = df[df['Company Name'].str.lower() == "swiggy"]

if swiggy_df.empty:
    st.error("No reviews found for Swiggy in the dataset.")
    st.stop()

st.subheader(f"Total Reviews for Swiggy: {len(swiggy_df)}")
st.dataframe(swiggy_df.head())

# Sentiment Analysis
def get_sentiment(text):
    analysis = TextBlob(str(text))
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

swiggy_df['Sentiment'] = swiggy_df['Review Text'].apply(get_sentiment)

# Sentiment distribution
st.subheader("Sentiment Distribution")
sentiment_counts = swiggy_df['Sentiment'].value_counts()
fig, ax = plt.subplots()
sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'], ax=ax)
plt.xticks(rotation=0)
st.pyplot(fig)

# Wordcloud
st.subheader("WordCloud of Reviews")
text = " ".join(swiggy_df['Review Text'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
ax_wc.imshow(wordcloud, interpolation='bilinear')
ax_wc.axis("off")
st.pyplot(fig_wc)

# Filter by rating
st.subheader("Filter Reviews by Rating")
rating_filter = st.slider("Select Rating", min_value=1, max_value=5, value=5)
filtered_df = swiggy_df[swiggy_df['Rating'] == rating_filter]
st.write(f"Showing {len(filtered_df)} reviews with {rating_filter}-star rating:")
st.dataframe(filtered_df[['Review Text', 'Rating', 'Sentiment']])
