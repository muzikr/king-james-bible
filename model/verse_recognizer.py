import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

bible_df = pd.read_csv(
    "../data/bible.txt",
    sep="\t",
    header=None,
    names=["ref", "text"],
    engine="python",
)
parts = bible_df["ref"].str.extract(
    r"^(?P<book>.+?)\s+(?P<chapter>\d+):(?P<verse>\d+)$"
)

bible_df = pd.concat([parts, bible_df["text"]], axis=1)

def assign_testament(book):
    old_testament_books = [
        # Historical books
        "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel", "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles", "Ezra", "Nehemiah", "Esther",
        # Poetic books
        "Job", "Psalms", "Proverbs", "Ecclesiastes", "Song of Solomon",
        # Prophetic books
        "Isaiah", "Jeremiah", "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos", "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi"
    ]

    new_testament_books = [
        # Gospels:
        "Matthew", "Mark", "Luke", "John",
        # Historical:
        "Acts",
        # Epistles:
        "Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians", "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews", "James", "1 Peter", "2 Peter", "1 John", "2 John", "3 John", "Jude",
        # The Revelation:
        "Revelation"
    ]

    if book in old_testament_books:
        return "Old Testament"
    elif book in new_testament_books:
        return "New Testament"
    else:
        return "Unknown"

bible_df["testament"] = bible_df["book"].apply(assign_testament)

import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

bible_df["cleaned_text"] = bible_df["text"].apply(clean_text)

with open("../data/stopwords-big.txt", "r") as f:
    stopwords = set(f.read().splitlines())

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import numpy as np
import copy

X_train, X_test, y_train, y_test = train_test_split(
    bible_df["cleaned_text"],
    bible_df["testament"],
    test_size=0.2,
    stratify=bible_df["testament"],
    random_state=42
)

# Dictionary to store results
results = {}

# Test different TfidfVectorizer and stopword configurations
configurations = [
    {
        'name': 'With stopwords, unigram+bigram',
        'stop_words': list(stopwords),
        'ngram_range': (1, 2),
        'min_df': 3,
        'max_features': 50000
    },
    {
        'name': 'Without stopwords, unigram+bigram',
        'stop_words': None,
        'ngram_range': (1, 2),
        'min_df': 3,
        'max_features': 50000
    },
    {
        'name': 'With stopwords, unigram only',
        'stop_words': list(stopwords),
        'ngram_range': (1, 1),
        'min_df': 3,
        'max_features': 50000
    },
    {
        'name': 'Without stopwords, unigram only',
        'stop_words': None,
        'ngram_range': (1, 1),
        'min_df': 3,
        'max_features': 50000
    },
    {
        'name': 'With stopwords, trigrams',
        'stop_words': list(stopwords),
        'ngram_range': (1, 3),
        'min_df': 5,
        'max_features': 50000
    }
]

# Models to test
model_specs = [
    ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
    ('Logistic Regression (L1)', LogisticRegression(max_iter=1000, penalty='l1', solver='liblinear', random_state=42)),
    ('Linear SVM', LinearSVC(max_iter=2000, random_state=42, dual=False)),
    ('Naive Bayes', MultinomialNB()),
]

best_score = 0
best_model_name = None
best_config_name = None
best_model = None
best_tfidf = None

print("=" * 80)
print("HYPERPARAMETER TUNING AND MODEL COMPARISON")
print("=" * 80)

for config in configurations:
    print(f"\n{'='*80}")
    print(f"Configuration: {config['name']}")
    print(f"{'='*80}")
    
    tfidf = TfidfVectorizer(
        ngram_range=config['ngram_range'],
        analyzer="word",
        min_df=config['min_df'],
        max_features=config['max_features'],
        stop_words=config['stop_words']
    )
    
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    print(f"Features created: {X_train_tfidf.shape[1]}")
    
    for model_name, model_template in model_specs:
        # Create fresh instance for this run
        model = copy.deepcopy(model_template)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring='accuracy')
        
        # Train and evaluate
        model.fit(X_train_tfidf, y_train)
        pred = model.predict(X_test_tfidf)
        test_accuracy = accuracy_score(y_test, pred)
        
        # ROC-AUC score (for binary classification)
        try:
            pred_proba = model.predict_proba(X_test_tfidf)
            roc_auc = roc_auc_score(y_test, pred_proba[:, 1], labels=[0, 1])
        except:
            roc_auc = None
        
        result_key = f"{config['name']} + {model_name}"
        results[result_key] = {
            'model': copy.deepcopy(model),
            'tfidf': copy.deepcopy(tfidf),
            'cv_score': cv_scores.mean(),
            'test_accuracy': test_accuracy,
            'roc_auc': roc_auc
        }
        
        print(f"\n{model_name}:")
        print(f"  Cross-validation score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"  Test accuracy: {test_accuracy:.4f}")
        if roc_auc is not None:
            print(f"  ROC-AUC score: {roc_auc:.4f}")
        
        # Update best model
        if test_accuracy > best_score:
            best_score = test_accuracy
            best_model_name = model_name
            best_config_name = config['name']
            best_model = copy.deepcopy(model)
            best_tfidf = copy.deepcopy(tfidf)

print(f"\n\n{'='*80}")
print("BEST MODEL FOUND")
print(f"{'='*80}")
print(f"Configuration: {best_config_name}")
print(f"Model: {best_model_name}")
print(f"Test Accuracy: {best_score:.4f}")
print(f"\nDetailed Classification Report:")
# Recreate the test data with the best tfidf
X_test_best = best_tfidf.transform(X_test)
pred_best = best_model.predict(X_test_best)
print(classification_report(y_test, pred_best))

# Save the best model and vectorizer
import pickle

with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('best_tfidf.pkl', 'wb') as f:
    pickle.dump(best_tfidf, f)

print(f"\n{'='*80}")
print("Model and vectorizer saved successfully!")
print(f"Best model: {best_model_name}")
print(f"Best configuration: {best_config_name}")
print(f"{'='*80}")
