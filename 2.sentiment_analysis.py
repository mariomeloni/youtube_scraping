import pandas as pd
from textblob import TextBlob



# Load the CSV file into a DataFrame
csv_file_path = "youtube_comments.csv"
df = pd.read_csv(csv_file_path)

# Define a function to get the sentiment label of a comment
def get_sentiment_label(polarity):
    if polarity > 0.2:
        return "Positive"
    elif polarity < -0.2:
        return "Negative"
    else:
        return "Neutral"

# Define a function to get the sentiment polarity and label of a comment
def get_sentiment(comment):
    # Perform sentiment analysis using TextBlob
    blob = TextBlob(comment)
    polarity = blob.sentiment.polarity
    label = get_sentiment_label(polarity)
    return polarity, label

# Apply the sentiment analysis function to each comment
df["Sentiment_Polarity"], df["Sentiment_Label"] = zip(*df["Comment"].apply(get_sentiment))

# Display the DataFrame with sentiment scores and labels
print(df)
df.to_csv("youtube_comments_sentiment.csv", index=True)


