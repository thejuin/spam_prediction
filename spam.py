# email_spam_app.py

import pandas as pd
import re
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Email Spam Detector", layout="centered")
st.title("📧 Email Spam Detection App")

# Load dataset
df = pd.read_csv("email.csv")

# Preview
st.subheader("Sample of Dataset")
st.write(df.head())

# Rename columns if needed
if df.columns[0].lower() != 'label' or df.columns[1].lower() != 'text':
    df.columns = ['label', 'text']

# Normalize label values
df['label'] = df['label'].astype(str).str.lower().map({'spam': 1, 'ham': 0})

# Drop any rows where label mapping failed (NaN)
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Apply cleaning
df['clean_text'] = df['text'].apply(clean_text)

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# Train-test split with stratify to preserve label distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.success(f"✅ Model trained with {accuracy * 100:.2f}% accuracy!")

# User Input Section
st.subheader("📨 Enter an Email to Predict")

user_input = st.text_area("Type or paste an email message here:")

if st.button("Predict"):
    cleaned_input = clean_text(user_input)
    vectorized_input = vectorizer.transform([cleaned_input])
    prediction = model.predict(vectorized_input)[0]

    if prediction == 1:
        st.error("🚨 This looks like SPAM!")
    else:
        st.success("✅ This looks like a LEGITIMATE (HAM) email.")
