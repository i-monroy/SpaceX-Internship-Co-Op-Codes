# Spam Detection Algorithm

## Author
Isaac Monroy

## Project Description
The algorithm utilizes Natural Language Processing (NLP) by implementing the Bag of Words technique using Count Vectorizer, and classifying messages using the Multinomial Naive Bayes classifier. It's trained on an SMS Spam Collection dataset to correctly classify spam messages from legitimate (ham) messages.

## Libraries Used
- **pandas:** For loading and manipulating the dataset.
- **numpy:** For numerical operations.
- **sklearn.model_selection:** For splitting the data into training and testing sets.
- **sklearn.feature_extraction.text:** For implementing the Bag of Words technique.
- **sklearn.naive_bayes:** For the Naive Bayes classification.
- **sklearn.metrics:** For evaluating the model's accuracy and confusion matrix.

## How to Run
1. Install the required libraries.
2. Load the SMS Spam Collection dataset.
3. Run the script to train the model, test it, and predict on new messages.

## Input and Output
- **Input:** The SMS messages to be classified.
- **Output:** Classification of the messages as spam or ham (legitimate), along with the model's accuracy and confusion matrix.
