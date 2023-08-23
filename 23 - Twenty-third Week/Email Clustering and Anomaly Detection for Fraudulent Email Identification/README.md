# Email Clustering and Anomaly Detection for Fraudulent Email Identification

## Author
Isaac Monroy

## Project Description
This code processes a dataset containing emails, extracting their content and relevant metadata. After cleaning and transforming the email content into numerical representations via TF-IDF vectorization, PCA is used for dimensionality reduction. The reduced feature set is then subjected to K-means clustering to segment the emails into distinct groups. The Elbow method aids in determining the optimal number of clusters. Post clustering, the script identifies and visualizes the top features characterizing each cluster. Finally, to detect potentially anomalous or fraudulent emails, the script calculates the distance of each email to its respective cluster centroid and employs an Isolation Forest model on the PCA-reduced feature set. Emails flagged as anomalies are then extracted and analyzed to identify their features.

## Libraries Used
- **os**: Handling files, directories, and OS-specific operations.
- **re**: Provides regex operations for advanced string manipulation and searching.
- **email**: Facilitates operations and manipulations on email messages.
- **pandas**: Enables efficient handling and operations on structured data with DataFrames.
- **numpy**: Provided support for numerical operations, arrays, and matrices.
- **matplotlib**: Plotting graphs and visualizations.
- **nltk**: Provides a list of stopwords, aiding in text preprocessing.
- **sklearn**: Implements various machine learning algorithms and preprocessing techniques.

## How to Run
1. Ensure all required libraries are installed.
2. Load your dataset of emails into the variable `df`.
3. Modify the path in `pd.read_csv('path/to/emails.csv')` to point to your dataset.
4. Run the entire script.

## Input and Output
- **Input**: A CSV file containing raw email data.
- **Output**: Clusters of similar emails, visualizations of clusters, top features of each cluster, potential anomalies within the clusters, high-risk emails, and visualization of the top keywords in high-risk emails.
