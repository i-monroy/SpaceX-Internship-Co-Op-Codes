"""
Author: Isaac Monroy
Project Title: Supervised Machine Learning Comparison
Description:
    This project illustrates the differences between various supervised 
    machine learning models by evaluating their performance on different 
    datasets. It includes interactive plots for visualizing how the models 
    behave with varying parameters, demonstrating their strengths and 
    weaknesses.
"""

# Import general modules
import numpy as np # For numerical operations.
import matplotlib.pyplot as plt # For plotting and 3D visualization.
import mglearn # For extended visualization of decision boundaries.
from ipywidgets import interact # For creating interactive widgets.
from sklearn.model_selection import train_test_split # To split the dataset into training and test sets.

# Import specific machine learning models and datasets
from sklearn.datasets import load_breast_cancer, make_blobs, make_moons # For loading predefined datasets.
from sklearn.tree import DecisionTreeClassifier # For Decision Tree classification.
from sklearn.ensemble import RandomForestClassifier # For Random Forest classification.
from sklearn.ensemble import GradientBoostingClassifier # For Gradient Boosting classification.
from sklearn.svm import LinearSVC, SVC # For Support Vector Machine classification.
from sklearn.neural_network import MLPClassifier # For Multi-Layer Perceptron (Neural Network) classification.

# Load breast cancer dataset
cancer = load_breast_cancer()

# Function to display feature importance for a model
def feature_importance(model):
    n_features = cancer.data.shape[1]
    cancer_f_names = cancer.feature_names
    cancer_f_names.sort()
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer_f_names)
    plt.title("Breast Cancer Dataset")
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    
# Function to illustrate the Decision Tree model
@interact
def tree_important_feat_plot(max_depth=(1,8,1)):
    X_tree_train, X_tree_test, y_tree_train, y_tree_test = train_test_split(
     cancer.data, cancer.target, stratify=cancer.target, random_state=42)

    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    tree.fit(X_tree_train, y_tree_train)

    print("Accuracy on training set: {:.3f}".format(tree.score(X_tree_train, y_tree_train)))
    print("Accuracy on test set: {:.3f}".format(tree.score(X_tree_test, y_tree_test)))
    feature_importance(tree)
    
# Function to illustrate the Random Forest model
@interact
def forest_important_feat_plot(n_estimators=(90,110,1)):
    X_forest_train, X_forest_test, y_forest_train, y_forest_test = train_test_split(
     cancer.data, cancer.target, random_state=0)

    forest = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
    forest.fit(X_forest_train, y_forest_train)

    print("Accuracy on training set: {:.3f}".format(forest.score(X_forest_train, y_forest_train)))
    print("Accuracy on test set: {:.3f}".format(forest.score(X_forest_test, y_forest_test)))
    feature_importance(forest)

# Function to illustrate the Gradient Booster Regression Trees model
@interact
def gbrt_important_feat_plot(max_depth=(1,4,1)):
    X_gbrt_train, X_gbrt_test, y_gbrt_train, y_gbrt_test = train_test_split(
     cancer.data, cancer.target, random_state=0)

    gbrt = GradientBoostingClassifier(max_depth=max_depth, random_state=0)
    gbrt.fit(X_gbrt_train, y_gbrt_train)

    print("Accuracy on training set: {:.3f}".format(gbrt.score(X_gbrt_train, y_gbrt_train)))
    print("Accuracy on test set: {:.3f}".format(gbrt.score(X_gbrt_test, y_gbrt_test))) 
    feature_importance(gbrt)

# Generate 4 Gaussian blobs
X, y = make_blobs(centers=4, random_state=8)
y = y % 2

# Instantiate model and train it
linear_svm = LinearSVC()
linear_svm.fit(X,y)

# Function to illustrate the Linear SVM model
@interact
def plot_2d_decision_boundary(decision_boundary = False):
    if decision_boundary:
        mglearn.plots.plot_2d_separator(linear_svm, X)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")

# Function to illustrate the 3D Linear SVM model
@interact
def plot_3d_decision_boundary(decision_boundary = False):
    # Instantiate model and train it
    X_new = np.hstack([X, X[:, 1:] ** 2])
    linear_svm_3d = LinearSVC()
    linear_svm_3d.fit(X_new, y)
    
    # Obtain coefficient and intercept for decision boundary
    coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_
    figure = plt.figure()
    ax = Axes3D(figure, elev=-152, azim=-26)
    
    # Plot decision boundary on 3d space
    if decision_boundary:
        xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
        yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)
        XX, YY = np.meshgrid(xx, yy)
        ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
        ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
        
    # Plot the blobs on the 3d space
    mask = y == 0
    ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
               cmap=mglearn.cm2, s=60, edgecolor='k')
    ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^',
               cmap=mglearn.cm2, s=60, edgecolor='k')
    ax.set_xlabel("feature0")
    ax.set_ylabel("feature1")
    ax.set_zlabel("feature1 ** 2")

# Load dataset
X, y = mglearn.tools.make_handcrafted_dataset()

# Function to illustrate the Radial Basis Function SVM model
@interact
def plot_decision_bound(C=[0.1, 1, 1000], gamma=[0.1, 1, 10]):
    svm = SVC(kernel='rbf', C=C, gamma=gamma)
    svm.fit(X,y)
    mglearn.plots.plot_2d_separator(svm, X, eps=.5)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    # Obtain support vectors
    sv = svm.support_vectors_
    # Obtain labels of support vectors
    sv_labels = svm.dual_coef_.ravel() > 0
    mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")

# Initialize varying variables
act_val = ['relu' ,'tanh']
hid_val = [[10, 10] ,[100, 100]]
a_val = [0.0001, 0.01, 0.1, 1]

# Function to illustrate the MLPClassifier NN model
@interact
def nn_plot(activation = act_val, hidden_layer_sizes = hid_val, alpha = a_val):
    X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    mlp = MLPClassifier(solver='lbfgs', random_state=0, activation = activation,
                        hidden_layer_sizes = hidden_layer_sizes,
                        alpha = alpha)
    mlp.fit(X_train, y_train)

    mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
