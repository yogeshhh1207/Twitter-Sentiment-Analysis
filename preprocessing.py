import re
import pandas as pd
from nltk.corpus import stopwords
import spacy
import os
import subprocess

# Downloading the spaCy English model
subprocess.check_call(["python3", "-m", "spacy", "download", "en_core_web_sm"])

BASE_DIR = os.getcwd()

# Loading the datasets
df_train = pd.read_csv(os.path.join(BASE_DIR, "database/train.csv"))
df_test = pd.read_csv(os.path.join(BASE_DIR, "database/test.csv"))

# Load spaCy English model
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Define stopwords
stop_words = set(stopwords.words('english'))

# Preprocessing steps
def preprocess_tweet(tweet):
    # Convert text to lowercase
    tweet = tweet.lower()
    # Remove emojis and punctuation
    tweet = re.sub(r'[^\w\s]', '', tweet)
    # Remove excess spaces
    tweet = re.sub(r'\s+', ' ', tweet)
    # Remove stopwords and lemmatize
    doc = nlp(tweet)
    lemmatized_tokens = [token.lemma_ for token in doc if token.text not in stop_words]
    return ' '.join(lemmatized_tokens)

# Applying preprocessing to tweet data
df_train['processed_tweet'] = df_train['tweet'].apply(preprocess_tweet)
df_test['processed_tweet'] = df_test['tweet'].apply(preprocess_tweet)

# Printing processed tweets
print(df_train[['tweet', 'processed_tweet']].head())
print('\n')
print(df_test[['tweet', 'processed_tweet']].head())

# Copy 'processed_tweet' column to 'tweet' column
df_train['tweet'] = df_train['processed_tweet']
df_test['tweet'] = df_test['processed_tweet']

# Drop the 'processed_tweet' column
df_train.drop(columns=['processed_tweet'], inplace=True)
df_test.drop(columns=['processed_tweet'], inplace=True)

# Save the final dataframes to CSV files
df_train.to_csv(os.path.join(BASE_DIR, "database/train.csv"), index=False)
df_test.to_csv(os.path.join(BASE_DIR, "database/test.csv"), index=False)
