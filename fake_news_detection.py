import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Only needed once — downloads stopwords
nltk.download('stopwords')

# Loading Dataset
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

# Add labels: 0 = Fake, 1 = Real
fake_df['label'] = 0
true_df['label'] = 1

# Combine and shuffle the data
data = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Keep only the necessary columns
data = data[['text', 'label']]

# Set of stopwords from NLTK
stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Apply cleaning to all news articles
data['cleaned_text'] = data['text'].apply(clean_text)

# Features (X) and Labels (y)
X = data['cleaned_text']
y = data['label']

# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=5000)

# Transform text to numerical format
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Initialize and train model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predict on test data
y_pred = model.predict(X_test_tfidf)

# Accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Detailed report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and TF-IDF vectorizer
joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("\nModel and vectorizer saved successfully.")

