# https://scipy-lectures.org/advanced/image_processing/index.html#basic-manipulations
import numpy as np
from scipy import stats, datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, fetch_openml
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.datasets import load_digits, load_iris, load_diabetes, load_wine


def task000():
    # Generate two random samples with different means
    np.random.seed(0)  # To make the result reproducible
    sample1 = np.random.normal(loc=5, scale=1, size=100)  # Sample 1 with mean 5 and standard deviation 1
    sample2 = np.random.normal(loc=7, scale=1, size=100)  # Sample 2 with mean 7 and standard deviation 1

    # Perform t-test
    t_statistic, p_value = stats.ttest_ind(sample1, sample2)

    # Show the results
    print("Test t statistic:", t_statistic)
    print("Associated p-value:", p_value)

    # Interpret the result
    alpha = 0.05
    if p_value < alpha:
        print("The difference between the means of the two groups is statistically significant (reject H0)")
    else:
        print("There is not enough evidence to say that the means of the two groups are different (fail to reject H0)")


def task001():
    '''
    This function loads the Digits dataset, visualizes some images, and provides basic statistics.
    '''

    # Load the Digits dataset
    digits = load_digits()

    # Visualize some images
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))
    for ax, image, label in zip(axes.flat, digits.images, digits.target):
        ax.imshow(image, cmap='gray')
        ax.set_title(f"Label: {label}")
        ax.axis('off')
    plt.show()

    # Print some basic statistics
    print("Number of samples:", len(digits.data))
    print("Number of features per sample:", len(digits.data[0]))
    print("Number of classes:", len(digits.target_names))
    print("Class distribution:")
    for i in range(len(digits.target_names)):
        print(f"Class {i}: {sum(digits.target == i)} samples")


def task002():
    '''
    This function performs a simple classification task using scikit-learn on the Digits dataset.
    '''

    # Load the Digits dataset
    digits = load_digits()

    # Split the dataset into features (X) and target (y)
    X = digits.data
    y = digits.target

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Support Vector Classifier
    clf = SVC()

    # Train the classifier using the training set
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = clf.predict(X_test)

    # Calculate the accuracy of the predictions
    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy:", accuracy)

def task003():
    '''
    This function performs clustering using scikit-learn on a synthetic blobs dataset.
    '''

    # Generate synthetic blobs dataset
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

    # Initialize K-Means clustering model
    kmeans = KMeans(n_clusters=4, random_state=42)

    # Fit the model to the data
    kmeans.fit(X)

    # Get cluster labels
    cluster_labels = kmeans.labels_

    # Visualize the clustering result
    fig, ax = plt.subplots()

    scatter = ax.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Clustering Result')

    # Add legend
    legend = ax.legend(*scatter.legend_elements(), title="Clusters", loc="upper left")
    ax.add_artist(legend)

    plt.show()

def task004():
    '''
    This function demonstrates the use of Principal Component Analysis (PCA) using scikit-learn.
    '''

    # Load the Iris dataset
    iris = load_iris()

    # Extract features
    X = iris.data

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize PCA model
    pca = PCA(n_components=2)

    # Fit the model to the data
    pca.fit(X_scaled)

    # Transform the data to the new feature space
    X_pca = pca.transform(X_scaled)

    # Visualize the transformed data
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target, cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Iris Dataset')
    plt.show()

    # Explained variance ratio
    print("Explained Variance Ratio:", pca.explained_variance_ratio_)


def plot_roc_curve_logistic_regression(X_train, y_train, X_test, y_test):
    '''
    This function plots the ROC curve for the logistic regression model on the training and testing data.
    '''

    # Initialize and train logistic regression model
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    # Predict probabilities for positive class
    y_train_prob = log_reg.predict_proba(X_train)[:, 1]
    y_test_prob = log_reg.predict_proba(X_test)[:, 1]

    # Compute ROC curve and ROC area for training set
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
    roc_auc_train = auc(fpr_train, tpr_train)

    # Compute ROC curve and ROC area for testing set
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
    roc_auc_test = auc(fpr_test, tpr_test)

    # Plot ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr_train, tpr_train, color='darkorange', lw=lw, label=f'Training ROC curve (AUC = {roc_auc_train:.2f})')
    plt.plot(fpr_test, tpr_test, color='navy', lw=lw, label=f'Testing ROC curve (AUC = {roc_auc_test:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Logistic Regression')
    plt.legend(loc="lower right")
    plt.show()

def task005():
    '''
    https://www.pycodemates.com/2022/05/iris-dataset-classification-with-python.html?utm_content=cmp-true

    This function loads the Iris dataset, creates a scatter plot, preprocesses the data, trains a logistic regression model,
    and evaluates its performance on training and testing sets.
    '''

    # Load Iris dataset
    iris = load_iris()

    # Scatter Plot
    _, ax = plt.subplots()
    scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
    ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
    _ = ax.legend(
        scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
    )

    # Create DataFrame from Iris dataset
    iris = pd.DataFrame(
        data=np.c_[iris['data'], iris['target']],
        columns=iris['feature_names'] + ['target']
    )

    # Assign species names
    species = []
    for i in range(len(iris['target'])):
        if iris['target'][i] == 0:
            species.append("setosa")
        elif iris['target'][i] == 1:
            species.append('versicolor')
        else:
            species.append('virginica')
    iris['species'] = species

    # Group by species and show size
    iris.groupby('species').size()

    # Summary statistics of the Iris dataset
    iris.describe()

    # Drop target and species columns
    X = iris.drop(['target', 'species'], axis=1)

    # Select petal length and petal width features
    X = X.to_numpy()[:, (2, 3)]
    y = iris['target']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


    # Initialize and train logistic regression model
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    # Predictions on training set
    training_prediction = log_reg.predict(X_train)
    test_prediction = log_reg.predict(X_test)

    print("Precision, Recall, Confusion matrix, in training\n")

    # Precision and Recall scores for training set
    print(metrics.classification_report(y_train, training_prediction, digits=3))

    # Confusion matrix for training set
    print(metrics.confusion_matrix(y_train, training_prediction))

    print("Precision, Recall, Confusion matrix, in testing\n")

    # Precision and Recall scores for testing set
    print(metrics.classification_report(y_test, test_prediction, digits=3))

    # Confusion matrix for testing set
    print(metrics.confusion_matrix(y_test, test_prediction))



def task006():
    '''
    This function loads the Credit Card Fraud Detection dataset, preprocesses the data, trains a logistic regression
    model and plots the ROC curve.
    '''

    # Carica il dataset Credit Card Fraud Detection
    data = fetch_openml(name='creditcard', version=1)

    # Divide i dati in feature e target
    X = data.data
    y = data.target

    # Dividi i dati in set di training e di testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardizza le feature
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Plot della curva ROC per la regressione logistica
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    plot_roc_curve_logistic_regression(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    # Basic Operation
    #task000()
    #input("\nNext Task ")

    #task001()
    #input("\nNext Task ")

    #task002()
    #input("\nNext Task ")

    #task003()
    #input("\nNext Task ")

    task004()
    input("\nNext Task ")

    task005()
    input("\nNext Task ")

    task006()
    input("\nNext Task ")