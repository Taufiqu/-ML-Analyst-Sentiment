import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load data
df = pd.read_csv('data/ulasan_900.csv')

# Split data
X = df['review']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline: TF-IDF + Naive Bayes
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# Train model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'models/sentiment_model.pkl')

# Optional: print accuracy
acc = model.score(X_test, y_test)
print(f"Accuracy: {acc:.2f}")
