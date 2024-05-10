import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import matplotlib.pyplot as plt
from nltk.text import Text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

csv_file_path = "../youtube_comments_sentiment.csv"
df = pd.read_csv(csv_file_path)

# Combine all comments into a single string (corpus)
corpus = " ".join(df["Comment"])

# Tokenize the corpus into words
tokens = word_tokenize(corpus)

# Remove stop words
stop_words = set(stopwords.words("english"))
filtered_tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]


# Find the most recurring bigrams
bigram_measures = BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(filtered_tokens)
bigrams = finder.nbest(bigram_measures.raw_freq, 10)

# Print the most recurring bigrams
print("\nTop 10 Most Recurring Bigrams:")
for bigram in bigrams:
    print(bigram)


# Calculate the frequency distribution of words
fdist = nltk.FreqDist(filtered_tokens)
plt.figure(figsize=(10, 6))
fdist.plot(30, cumulative=False)
plt.title("Frequency Distribution of Words")
plt.xlabel("Word")
plt.ylabel("Frequency")
plt.show()

# Print the top 10 most common words
print("Top 10 Most Common Words:")
print(fdist.most_common(10))

# concordance analysis
# Convert the corpus into NLTK Text object
text = Text(filtered_tokens)

# Perform concordance analysis for a specific word
word_to_concord = "better"
concordance_results = text.concordance(word_to_concord, width=100, lines=10)
print(concordance_results)


# topic modeling
comments = df["Comment"].tolist()

# Create a document-term matrix using CountVectorizer
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = vectorizer.fit_transform(comments)

# Apply Latent Dirichlet Allocation (LDA)
num_topics = 5
lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda_model.fit(dtm)

# Display the topics
for idx, topic in enumerate(lda_model.components_):
    print(f"Topic {idx}:")
    top_words_idx = topic.argsort()[-10:]
    top_words = [vectorizer.get_feature_names_out()[i] for i in top_words_idx]
    print(top_words)
    print()