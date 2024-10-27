import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import spacy
from spacy.tokens import Doc
from spacy.matcher import PhraseMatcher, Matcher
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import re
from collections import defaultdict
import json
import os
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimelineTracker:
    def __init__(self):
        self.timeline = []
        
    def add_entry(self, analysis_result):
        """Add a new analysis entry to the timeline"""
        self.timeline.append(analysis_result)
        
    def get_sentiment_progression(self, days=30):
        """Analyze sentiment progression over time"""
        if not self.timeline:
            return None
            
        progression = []
        for entry in self.timeline[-days:]:
            # Check if there are any concerns before calculating mean
            concerns = [c['category'] for c in entry['concerns']]
            intensities = [c['intensity'] for c in entry['concerns']]
            
            # Calculate average intensity only if there are concerns
            avg_intensity = np.mean(intensities) if intensities else 0
            
            progression.append({
                'timestamp': entry['timestamp'],
                'polarity': entry['polarity']['dominant_polarity'],
                'confidence': entry['polarity']['confidence'],
                'concerns': concerns,
                'average_intensity': float(avg_intensity)  # Convert to float to handle np.nan
            })
            
        return progression
        
    def detect_shifts(self):
        """Detect significant shifts in mental health states"""
        if len(self.timeline) < 2:
            return None
            
        shifts = []
        for i in range(1, len(self.timeline)):
            prev = self.timeline[i-1]
            curr = self.timeline[i]
            
            # Check for polarity shifts
            if prev['polarity']['dominant_polarity'] != curr['polarity']['dominant_polarity']:
                shifts.append({
                    'type': 'polarity_shift',
                    'from': prev['polarity']['dominant_polarity'],
                    'to': curr['polarity']['dominant_polarity'],
                    'timestamp': curr['timestamp']
                })
            
            # Check for category shifts
            prev_categories = set(c['category'] for c in prev['concerns'])
            curr_categories = set(c['category'] for c in curr['concerns'])
            
            new_categories = curr_categories - prev_categories
            if new_categories:
                shifts.append({
                    'type': 'new_concerns',
                    'categories': list(new_categories),
                    'timestamp': curr['timestamp']
                })
            
            # Check for intensity shifts - only if both entries have concerns
            prev_intensities = [c['intensity'] for c in prev['concerns']]
            curr_intensities = [c['intensity'] for c in curr['concerns']]
            
            if prev_intensities and curr_intensities:  # Only calculate if both have intensities
                prev_avg_intensity = np.mean(prev_intensities)
                curr_avg_intensity = np.mean(curr_intensities)
                
                if abs(curr_avg_intensity - prev_avg_intensity) >= 2:
                    shifts.append({
                        'type': 'intensity_shift',
                        'from': float(prev_avg_intensity),  # Convert to float to handle np.nan
                        'to': float(curr_avg_intensity),    # Convert to float to handle np.nan
                        'timestamp': curr['timestamp']
                    })
                
        return shifts

class MentalHealthAnalyzer:
    """Main class for mental health text analysis"""
    
    def __init__(self, dataset_path):
        """Initialize the analyzer with models and data"""
        try:
            # Initialize timeline tracker first
            self.timeline_tracker = TimelineTracker()
            
            logger.info("Loading pre-trained models...")
            # Load pre-trained models
            self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
            self.nlp = spacy.load("en_core_web_sm")
            
            logger.info("Loading dataset...")
            # Load and prepare dataset
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
                
            self.df = pd.read_excel(dataset_path)
            self.df.columns = self.df.columns.str.strip()
            
            required_columns = ['User Input', 'Category', 'Intensity', 'Extracted Concern']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in dataset: {missing_columns}")
            
            logger.info("Initializing models...")
            self.initialize_models()
            
            logger.info("Creating matchers...")
            self.create_matchers()
            
            logger.info("System initialization complete.")
            
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            raise

    def create_matchers(self):
        """Create pattern matchers for keyword extraction"""
        try:
            # Initialize matchers
            self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
            self.matcher = Matcher(self.nlp.vocab)
            
            # Create patterns from training data
            concern_patterns = self.df['Extracted Concern'].dropna().unique()
            patterns = [self.nlp(text) for text in concern_patterns]
            self.phrase_matcher.add("MENTAL_HEALTH_CONCERNS", patterns)
            
            # Add linguistic patterns
            self.matcher.add("SEVERITY_PATTERNS", [
                # Extreme severity patterns
                [{"LOWER": {"IN": ["extremely", "severely", "terribly"]}},
                 {"LOWER": {"IN": ["anxious", "depressed", "worried", "stressed", "sad"]}}],
                
                # High severity patterns
                [{"LOWER": {"IN": ["very", "really", "so"]}},
                 {"LOWER": {"IN": ["anxious", "depressed", "worried", "stressed", "sad"]}}],
                
                # Medium severity patterns
                [{"LOWER": {"IN": ["quite", "somewhat", "fairly"]}},
                 {"LOWER": {"IN": ["anxious", "depressed", "worried", "stressed", "sad"]}}],
                
                # Low severity patterns
                [{"LOWER": {"IN": ["slightly", "a bit", "mildly"]}},
                 {"LOWER": {"IN": ["anxious", "depressed", "worried", "stressed", "sad"]}}],
            ])
            
            # Add personal expression patterns
            self.matcher.add("PERSONAL_EXPRESSIONS", [
                [{"LOWER": "i"}, 
                 {"LOWER": {"IN": ["feel", "am", "think"]}},
                 {"OP": "?"}, 
                 {"LOWER": {"IN": ["anxious", "depressed", "worried", "stressed", "sad"]}}],
                [{"LOWER": "i"}, 
                 {"LOWER": "can't"}, 
                 {"LOWER": {"IN": ["sleep", "eat", "focus", "work"]}}],
            ])
            
        except Exception as e:
            logger.error(f"Error creating matchers: {str(e)}")
            raise

    def initialize_models(self):
        """Initialize and train classification models"""
        try:
            # Prepare training data
            X = self.df['User Input'].values
            y_category = self.df['Category'].values
            y_intensity = self.df['Intensity'].values
            
            # Create and fit vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 3),
                stop_words='english'
            )
            X_transformed = self.vectorizer.fit_transform(X)
            
            # Initialize and train category classifier
            self.category_encoder = LabelEncoder()
            y_category_encoded = self.category_encoder.fit_transform(y_category)
            self.category_classifier = RandomForestClassifier(n_estimators=100)
            self.category_classifier.fit(X_transformed, y_category_encoded)
            
            # Initialize and train intensity classifier
            self.intensity_classifier = RandomForestClassifier(n_estimators=100)
            self.intensity_classifier.fit(X_transformed, y_intensity)
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    def detect_polarity(self, text):
        """Detect emotional polarity of text"""
        try:
            encoded_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            output = self.sentiment_model(**encoded_input)
            scores = softmax(output[0][0].detach().numpy())
            
            # Map scores to labels
            labels = ['negative', 'neutral', 'positive']
            polarity_scores = {label: float(score) for label, score in zip(labels, scores)}
            
            # Get dominant polarity
            dominant_polarity = max(polarity_scores.items(), key=lambda x: x[1])
            
            return {
                'scores': polarity_scores,
                'dominant_polarity': dominant_polarity[0],
                'confidence': dominant_polarity[1]
            }
        except Exception as e:
            logger.error(f"Error detecting polarity: {str(e)}")
            return None

    def extract_keywords(self, text):
        """Extract mental health-related keywords from text"""
        try:
            doc = self.nlp(text)
            keywords = set()
            
            # Get phrase matcher matches
            phrase_matches = self.phrase_matcher(doc)
            for match_id, start, end in phrase_matches:
                span = doc[start:end]
                keywords.add(span.text)
            
            # Get pattern matcher matches
            pattern_matches = self.matcher(doc)
            for match_id, start, end in pattern_matches:
                span = doc[start:end]
                keywords.add(span.text)
                
            # Extract relevant noun chunks
            mental_health_terms = {
                "anxiety", "depression", "stress", "worry", "fear", 
                "panic", "trauma", "insomnia", "eating", "disorder"
            }
            for chunk in doc.noun_chunks:
                if any(term in chunk.text.lower() for term in mental_health_terms):
                    keywords.add(chunk.text)
            
            return list(keywords)
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []

    def classify_concern(self, text, keywords):
        """Classify mental health concerns and assign intensity scores"""
        try:
            # Vectorize input
            X = self.vectorizer.transform([text])
            
            # Get category prediction
            category_pred = self.category_classifier.predict(X)
            category = self.category_encoder.inverse_transform(category_pred)[0]
            
            # Get intensity prediction
            intensity_pred = self.intensity_classifier.predict(X)[0]
            
            # Adjust intensity based on linguistic cues
            intensity_modifiers = {
                'extremely': 2.0, 'severely': 2.0,
                'very': 1.5, 'really': 1.5,
                'quite': 1.0, 'somewhat': 0.8,
                'slightly': 0.5, 'a bit': 0.5
            }
            
            for keyword in keywords:
                for modifier in intensity_modifiers:
                    if modifier in keyword.lower():
                        intensity_pred *= intensity_modifiers[modifier]
                        break
            
            # Ensure intensity stays within 1-10 range
            intensity_pred = max(1, min(10, intensity_pred))
            
            return category, round(intensity_pred)
        except Exception as e:
            logger.error(f"Error classifying concern: {str(e)}")
            return None, None

    def analyze_text(self, text, timestamp=None):
        """Perform complete analysis of input text"""
        try:
            if timestamp is None:
                timestamp = datetime.now()
                
            # Get polarity
            polarity_results = self.detect_polarity(text)
            
            # Extract keywords
            keywords = self.extract_keywords(text)
            
            # Get category and intensity for each keyword
            concerns = []
            for keyword in keywords:
                category, intensity = self.classify_concern(keyword, keywords)
                if category and intensity:
                    concerns.append({
                        'keyword': keyword,
                        'category': category,
                        'intensity': intensity
                    })
            
            # Update timeline
            analysis_result = {
                'timestamp': timestamp,
                'text': text,
                'polarity': polarity_results,
                'keywords': keywords,
                'concerns': concerns
            }
            self.timeline_tracker.add_entry(analysis_result)
            
            return analysis_result
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            return None

def main():
    try:
        # Initialize analyzer
        dataset_path = os.path.join(os.path.dirname(__file__), 'mental_health_dataset.xlsx')
        analyzer = MentalHealthAnalyzer(dataset_path)
        
        print("Mental Health Analysis System")
        print("Enter text to analyze (or 'timeline' to see progression, 'exit' to quit)")
        
        while True:
            command = input("\nEnter command: ").strip()
            
            if command.lower() == 'exit':
                break
                
            elif command.lower() == 'timeline':
                # Show timeline analysis
                progression = analyzer.timeline_tracker.get_sentiment_progression()
                shifts = analyzer.timeline_tracker.detect_shifts()
                
                if progression:
                    print("\nSentiment Progression:")
                    for entry in progression:
                        print(f"\nDate: {entry['timestamp']}")
                        print(f"Polarity: {entry['polarity']}")
                        print(f"Concerns: {', '.join(entry['concerns'])}")
                        print(f"Average Intensity: {entry['average_intensity']:.1f}")
                    
                    if shifts:
                        print("\nDetected Shifts:")
                        for shift in shifts:
                            print(f"\nType: {shift['type']}")
                            if shift['type'] == 'polarity_shift':
                                print(f"Changed from {shift['from']} to {shift['to']}")
                            elif shift['type'] == 'new_concerns':
                                print(f"New concerns: {', '.join(shift['categories'])}")
                            elif shift['type'] == 'intensity_shift':
                                print(f"Intensity changed from {shift['from']:.1f} to {shift['to']:.1f}")
                else:
                    print("\nNo timeline data available yet.")
            elif command:  # Only process non-empty commands
                try:
                    results = analyzer.analyze_text(command)
                    
                    print("\nAnalysis Results:")
                    print("----------------")
                    
                    # Display polarity
                    print(f"Sentiment: {results['polarity']['dominant_polarity'].capitalize()}")
                    print(f"Confidence: {results['polarity']['confidence']:.2f}")
                    
                    # Display concerns
                    if results['concerns']:
                        print("\nDetected Concerns:")
                        for concern in results['concerns']:
                            print(f"- {concern['keyword']}")
                            print(f"  Category: {concern['category']}")
                            print(f"  Intensity: {concern['intensity']}/10")
                    else:
                        print("\nNo specific concerns detected.")
                    
                    # Show recent changes if available
                    shifts = analyzer.timeline_tracker.detect_shifts()
                    if shifts:
                        print("\nRecent Changes:")
                        for shift in shifts[-3:]:
                            if shift['type'] == 'polarity_shift':
                                print(f"- Mood shifted from {shift['from']} to {shift['to']}")
                            elif shift['type'] == 'intensity_shift':
                                direction = "increased" if shift['to'] > shift['from'] else "decreased"
                                print(f"- Overall intensity has {direction}")
                except Exception as e:
                    print(f"Error analyzing text: {str(e)}")
            else:
                print("Please enter some text to analyze.")
                
    except Exception as e:
        print(f"Error initializing the system: {str(e)}")
        print("Please ensure all required files and dependencies are properly installed.")

if __name__ == "__main__":
    main()