"""
Author: Isaac Monroy
Title: Spam Detection Algorithm
Description: 
    The algorithm utilizes NLP by using Count 
    Vectorizer, implementing the Bag of Words
    technique, and using MultinomialNB for the 
    Naive Bayes classifier. The model is trained
    on an SMS Spam Collection dataset so it can
    correctly classify spam messages from legit
    (ham) messages.
"""
# Import necessary modules
import pandas as pd # For data manipulation and analysis
import numpy as np # For numerical operations
from sklearn.model_selection import train_test_split # For splitting the dataset
from sklearn.feature_extraction.text import CountVectorizer # For implementing Bag of Words technique
from sklearn.naive_bayes import MultinomialNB # For Naive Bayes classification
from sklearn.metrics import accuracy_score, confusion_matrix # For evaluating the model

# Load data into a pandas dataframe
filename = 'SMSSpamCollection'
data = pd.read_csv(filename, sep='\t', names=['label', 'message'])

# Prepare the dataset
X = data['message']
y = data['label']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to vectors using CountVectorizer (Bag of Words)
vectorizer = CountVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)

# Train the Naive Bayes classifier with alpha=1.0
clf = MultinomialNB(alpha=1.0)
clf.fit(X_train_transformed, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test_transformed)

# Evaluate the model using accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion)

def predict_spam(message):
    """Predict whether a given message is spam or not"""
    message_transformed = vectorizer.transform([message])
    prediction = clf.predict(message_transformed)
    return prediction[0]

# Test the model with new messages
test_messages = [
    "Congratulations! You've won a $1,000 gift card. Click here to claim your prize.",
    "Hey, are you coming to the party tonight?",
    "Your account has been suspended. Please follow the link to reactivate your account.",
    "Don't forget to complete the homework for tomorrow's class."
]

for message in test_messages:
    print(f"Message: {message}")
    print(f"Prediction: {predict_spam(message)}\n")

