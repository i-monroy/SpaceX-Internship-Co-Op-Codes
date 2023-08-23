"""
Author: Isaac Monroy
Title: Email Clustering and Anomaly Detection for Fraudulent Email Identification
Description: 
    This code processes a dataset containing emails, extracting their content
    and relevant metadata. After cleaning and transforming the email content 
    into numerical representations via TF-IDF vectorization, a PCA is used for
    dimensionality reduction. The reduced feature set is then subjected to K-means
    clustering to segment the emails into distinct groups. The Elbow method aids
    in determining the optimal number of clusters. Post clustering, the script 
    identifies and visualizes the top features characterizing each cluster. Finally,
    to detect potentially anomalous or fraudulent emails, the script calculates the 
    distance of each email to its respective cluster centroid and employs an Isolation
    Forest model on the PCA-reduced feature set. Emails flagged as anomalies are 
    then extracted and analyzed to identify their features.
"""

import os # Handling files, directories, and OS-specific operations.
import re # Provides regex operations for advanced string manipulation and searching.
import email # Facilitates operations and manipulations on email messages.
import pandas as pd # Enables efficient handling and operations on structured data with DataFrames.
import numpy as np # Provided support for numerical operations, arrays, and matrices.
import matplotlib.pyplot as plt # Plotting graphs and visualizations.
from nltk.corpus import stopwords # Provides a list of stopwords, aiding in text preprocessing.
from sklearn.cluster import KMeans # Implements the K-means clustering algorithm.
from sklearn.decomposition import PCA # Conducts Principal Component Analysis for dimensionality reduction.
from sklearn.preprocessing import normalize # Used for normalizing data, making it suitable for machine learning algorithms.
from sklearn.feature_extraction.text import TfidfVectorizer # Converts text data into numerical form with Term Frequency-Inverse Document Frequency.
import email.parser # Used for parsing email structures and extracting headers.
from sklearn.ensemble import IsolationForest # Implements an ensemble-based anomaly detection method.

# Load the dataset
df = pd.read_csv('path/to/emails.csv',nrows = 35000)

# Extract email content from the 'message' column
emails = list(map(email.parser.Parser().parsestr, df['message']))

# Extract email metadata (e.g., 'From', 'To', 'Subject') 
for key in emails[0].keys():
    df[key] = [doc[key] for doc in emails]

def get_raw_text(email_obj):
    """
    Extract plain text content from an email object.
    """
    email_text = []
    for email_part in email_obj.walk():
        if email_part.get_content_type() == 'text/plain':
            email_text.append(email_part.get_payload())
    return ''.join(email_text)

# Apply the function to extract email content
df['body'] = list(map(get_raw_text, emails))

def clean_content(content):
    """
    Clean the email content by lowercasing all words, 
    removing non-alphanumeric characters, removing 
    numbers, and removing stopwords.
    """
    content = content.lower()
    content = re.sub(r'[^\w\s]', '', content)
    content = re.sub(r'\n', ' ', content)
    content = re.sub(r'[0-9]+', '', content)
    stopwords_list = stopwords.words('english')
    words = content.split()
    content = ' '.join([word for word in words if word not in stopwords_list])
    return content.strip()

# Apply cleaning function
df['body_cleaned'] = df['body'].apply(clean_content)

# Convert cleaned emails into a matrix of TF-IDF features
tf_idf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tf_idf = tf_idf_vectorizer.fit_transform(df['body_cleaned'])
tf_idf_norm = normalize(tf_idf)
tf_idf_array = tf_idf_norm.toarray()

# Reduce the dimensionality of the TF-IDF features to 2 using PCA
pca = PCA(n_components=2)
Y_pca = pca.fit_transform(tf_idf_array)

# Determine the optimal number of clusters using the Elbow method
n_clusters_range = range(1, 7)
kmeans_scores = [-KMeans(n_clusters=i).fit(Y_pca).score(Y_pca) for i in n_clusters_range]

plt.plot(n_clusters_range, kmeans_scores)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Method')
plt.show()

# Perform KMeans clustering on the PCA-reduced data
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, max_iter=600, algorithm='auto')
kmeans.fit(Y_pca)
prediction = kmeans.predict(Y_pca)

# Visualize the clusters
plt.scatter(Y_pca[:, 0], Y_pca[:, 1], c=prediction, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=300, alpha=0.6)
plt.show()

def get_top_features_cluster(tf_idf_array, prediction, n_feats):
    """
    Extract the top features for each cluster.
    """
    labels = np.unique(prediction)
    dfs = []
    for label in labels:
        id_temp = np.where(prediction==label)
        x_means = np.mean(tf_idf_array[id_temp], axis=0)
        sorted_means = np.argsort(x_means)[::-1][:n_feats]
        features = tf_idf_vectorizer.get_feature_names()
        best_features = [(features[i], x_means[i]) for i in sorted_means]
        df_temp = pd.DataFrame(best_features, columns=['features', 'score'])
        dfs.append(df_temp)
    return dfs

# Extract top features for each cluster
dfs = get_top_features_cluster(tf_idf_array, prediction, 20)

# Print top features for each cluster
for i, df_temp in enumerate(dfs):
    print(f"Cluster {i}:")
    print(df_temp)
    print("\n")

# Identify anomalies within the clusters
distances = kmeans.transform(Y_pca)
df['Distance_to_Centroid'] = np.min(distances, axis=1)
threshold = np.percentile(df['Distance_to_Centroid'], 95)
potential_outliers = df[df['Distance_to_Centroid'] > threshold]
print("Potential Outliers:")
print(potential_outliers)

# Apply Isolation Forest to detect anomalies in the PCA-reduced data
clf = IsolationForest(contamination=0.05)
preds = clf.fit_predict(Y_pca)
df_anomalies = pd.DataFrame(Y_pca, columns=['PC1', 'PC2'])
df_anomalies['anomaly'] = preds
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters).fit(Y_pca)
df_anomalies['cluster'] = kmeans.labels_

# Identify clusters with anomalies
high_risk_zones = df_anomalies[df_anomalies['anomaly'] == -1]['cluster'].unique()

# Extract emails from high-risk clusters
high_risk_emails = df[df_anomalies['cluster'].isin(high_risk_zones)]

# Identify top keywords in high-risk emails
tf_idf_high_risk = tf_idf_vectorizer.transform(high_risk_emails['body_cleaned'])
sum_words = tf_idf_high_risk.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in tf_idf_vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

# Visualize anomaly detection results
plt.figure(figsize=(10, 7))
colors = {1: 'blue', -1: 'red'}
plt.scatter(df_anomalies['PC1'], df_anomalies['PC2'], c=[colors[i] for i in df_anomalies['anomaly']])
plt.title('Anomaly Detection (Red points are detected anomalies)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Visualize top keywords in high-risk emails
top_n = 20
top_words = words_freq[:top_n]
labels, values = zip(*top_words)
plt.figure(figsize=(12, 8))
plt.barh(labels, values, color='skyblue')
plt.xlabel('Frequency')
plt.title(f'Top {top_n} Keywords in High-Risk Emails')
plt.gca().invert_yaxis()
plt.show()