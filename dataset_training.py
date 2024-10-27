import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from scipy.special import softmax
from transformers import AutoConfig
import spacy
from spacy.matcher import PhraseMatcher
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer  # Import CountVectorizer
import re
from collections import defaultdict
import os

# Load models
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
config = AutoConfig.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Load the mental health dataset
dataset_path = os.path.join(os.path.dirname(__file__), 'mental_health_dataset.xlsx')
df = pd.read_excel(dataset_path)

# Clean column names to remove any leading/trailing whitespace
df.columns = df.columns.str.strip()

# Prepare training data
training_data = [(row['User Input'], row['Category'], row['Intensity']) for index, row in df.iterrows()]

# Preprocess input text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove non-word characters
    return text.strip()

# Polarity Detection
def detect_polarity(text):
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return {config.id2label[i]: np.round(float(scores[i]), 4) for i in range(len(scores))}

# Enhanced Keyword Extractor
def extract_keywords(text):
    doc = nlp(text)
    keywords = []

    # Use PhraseMatcher to find mental health-related keywords
    matcher = PhraseMatcher(nlp.vocab)
    
    # Define a comprehensive list of mental health-related keywords
    mental_health_keywords = [
        "anxious", "depressed", "stress", "insomnia", "eating disorder", 
        "feeling low", "panic", "overwhelmed", "helpless", "numb", 
        "crying", "lost hope", "no one cares", "dark thoughts", 
        "self-harm", "suicidal", "extremely anxious", "constantly worried",
        "worried about health", "feel very anxious", "i want to die",
        "i hate", "i feel", "help me", "i am", "i can't", "i don't"
    ]
    
    patterns = [nlp(keyword) for keyword in mental_health_keywords]
    matcher.add("MentalHealthKeywords", patterns)

    matches = matcher(doc)
    for match_id, start, end in matches:
        keywords.append(doc[start:end].text)

    # Also consider noun chunks as potential keywords
    keywords += [chunk.text for chunk in doc.noun_chunks]

    return list(set(keywords))  # Return unique keywords

# Concern Classifier
def classify_concerns(keywords):
    # Prepare training data
    X_train = [text for text, label, _ in training_data]
    y_train = [label for _, label, _ in training_data]

    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    # Train a simple classifier
    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    classifier = MultinomialNB()
    classifier.fit(X_train_vectorized, y_train_encoded)

    # Classify extracted keywords
    classified_concerns = {}
    if keywords:  # Check if keywords are not empty
        keywords_vectorized = vectorizer.transform(keywords)
        predictions = classifier.predict(keywords_vectorized)
        classified_concerns = {keywords[i]: le.inverse_transform([predictions[i]])[0] for i in range(len(keywords))}
    
    return classified_concerns

# Intensity Scorer
def score_intensity(classified_concerns):
    intensity_scores = {}
    for concern in classified_concerns.values():
        # Find the corresponding intensity from the training data
        for _, label, intensity in training_data:
            if label == concern:
                intensity_scores[concern] = intensity
                break
    return intensity_scores

# Timeline-Based Sentiment Analyzer
class TimelineAnalyzer:
    def __init__(self):
        self.history = defaultdict(list)

    def add_entry(self, date, polarity_results, classified_concerns):
        self.history[date].append((polarity_results, classified_concerns))

    def analyze_timeline(self):
        analysis = {}
        for date, entries in self.history.items():
            avg_sentiment = np.mean([entry[0]['positive'] for entry in entries])
            analysis[date] = avg_sentiment
        return analysis

# Main execution loop
def main():
    # Train the classifier on the dataset
    print("Training the model on the dataset...")
    
    while True:
        user_input = input("Enter a line (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break

        # Preprocess the input text
        text = preprocess_text(user_input)

        # Detect polarity
        polarity_results = detect_polarity(text)

        # Extract keywords
        keywords = extract_keywords(text)

        # Classify concerns
        classified_concerns = classify_concerns(keywords)

        # Score intensity
        intensity_scores = score_intensity(classified_concerns)

        # Output results
        print("\nResults:")
        # print("Polarity Results:", polarity_results)
        if polarity_results['positive'] > polarity_results['negative'] :
            if polarity_results['positive'] > polarity_results['neutral']:
                print("Polarity: Positive")
            else:
                print("Polarity: Neutral")
        else:
            if polarity_results['negative'] > polarity_results['neutral']:
                print("Polarity: Negative")
            else:
                print("Polarity: Neutral")
        
        print("Extracted Keywords:", keywords)
        print("Classified Concerns:", classified_concerns)
        print("Intensity Scores:", intensity_scores)
        print()

if __name__ == "__main__":
    main()