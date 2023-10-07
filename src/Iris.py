import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Function to calculate the covariance matrix
def calculate_covariance_matrix(X):
    n_samples = X.shape[0]     # number of samples per row
    covariance_matrix = np.dot(X.T, X) / (n_samples - 1)
    return covariance_matrix


# Function to normalize the data
def normalize_data(X):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_normalized = (X - X_mean) / X_std
    return X_normalized


# Function to perform eigen decomposition
def perform_eigen_decomposition(covariance_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    eigenvalue_indices = np.argsort(eigenvalues)[::-1]           # sorts the eigenvalues index from the largest to smallest
    sorted_eigenvalues = eigenvalues[eigenvalue_indices]         # sorts the eigenvalues
    sorted_eigenvectors = eigenvectors[:, eigenvalue_indices]    # gets the eigenvectors based on each eigen value
    return sorted_eigenvalues, sorted_eigenvectors


# Function to project data onto the principal components
def project_data(X, eigenvectors, n_components):
    projected_data = np.dot(X, eigenvectors[:, :n_components])        # returns the principal components
    return projected_data


# Function to plot histograms of each feature colored by class label
def plot_histograms(X, y, feature_names):
    n_features = X.shape[1]
    n_classes = len(np.unique(y))

    fig, axs = plt.subplots(n_features, 1, figsize=(8, 6 * n_features))
    fig.subplots_adjust(hspace=0.4)

    for i in range(n_features):
        axs[i].set_title(f"Histogram of {feature_names[i]}")

        for label in np.unique(y):
            axs[i].hist(X[y == label, i], bins=20, alpha=0.5, label=str(label))

        axs[i].legend()
        axs[i].set_xlabel("Feature Value")
        axs[i].set_ylabel("Frequency")

    plt.show()


# Function to plot scatterplots of transformed data colored by class label
def plot_scatterplots(X, y):
    class_labels = np.unique(y)
    plt.figure()
    for label in class_labels:
        X_label = X[y == label]
        plt.scatter(X_label[:, 0], X_label[:, 1], label=str(label))
    plt.legend()
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Scatterplot of Transformed Data")
    plt.show()


# Function to build a decision tree classifier
class DecisionTreeClassifier:
    def __init__(self, max_depth=None, random_state=None):
        self.max_depth = max_depth
        self.random_state = random_state
        self.tree = None

    def _calculate_gini_index(self, y):                         # To know the best condition to split
        class_labels, class_counts = np.unique(y, return_counts=True)
        class_probabilities = class_counts / len(y)
        gini_index = 1 - np.sum(class_probabilities ** 2)
        return gini_index

    def _split_node(self, X, y, feature_index, threshold):     # Splits the data into left and right nodes
        left_mask = X[:, feature_index] <= threshold           # based on feature index (X0) and threshold
        right_mask = X[:, feature_index] > threshold
        left_X, left_y = X[left_mask], y[left_mask]
        right_X, right_y = X[right_mask], y[right_mask]
        return left_X, left_y, right_X, right_y

    def _find_best_split(self, X, y):
        best_gini = float('inf')
        best_feature_index = None
        best_threshold = None

        n_features = X.shape[1]
        for feature_index in range(n_features):
            feature_values = X[:, feature_index]
            unique_values = np.unique(feature_values)
            for threshold in unique_values:
                left_X, left_y, right_X, right_y = self._split_node(X, y, feature_index, threshold)
                left_gini = self._calculate_gini_index(left_y)
                right_gini = self._calculate_gini_index(right_y)
                gini = (len(left_y) * left_gini + len(right_y) * right_gini) / len(y)
                if gini < best_gini:
                    best_gini = gini
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def _create_leaf_node(self, y):
        class_labels, class_counts = np.unique(y, return_counts=True)
        majority_class_index = np.argmax(class_counts)
        leaf_node = {"class": class_labels[majority_class_index], "count": len(y)}
        return leaf_node

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth:
            return self._create_leaf_node(y)

        if len(np.unique(y)) == 1:
            return self._create_leaf_node(y)

        feature_index, threshold = self._find_best_split(X, y)
        if feature_index is None or threshold is None:
            return self._create_leaf_node(y)

        left_X, left_y, right_X, right_y = self._split_node(X, y, feature_index, threshold)

        node = {
            "feature_index": feature_index,
            "threshold": threshold,
            "left": self._build_tree(left_X, left_y, depth + 1),
            "right": self._build_tree(right_X, right_y, depth + 1),
        }

        return node

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _predict_single_sample(self, x):
        node = self.tree
        while "class" not in node:
            feature_index = node["feature_index"]
            threshold = node["threshold"]
            if x[feature_index] <= threshold:
                node = node["left"]
            else:
                node = node["right"]
        return node["class"]

    def predict(self, X):                       # Takes the samples and returns the classes that it belongs to
        n_samples = X.shape[0]
        y_pred = np.empty(n_samples, dtype=object)
        for i in range(n_samples):
            y_pred[i] = self._predict_single_sample(X[i])
        return y_pred


# Load the data from a CSV file
data = pd.read_csv('IRIS.csv')

# Shuffle the data
shuffled_data = data.sample(frac=1, random_state=110)

# Extract the feature data and target labels from the shuffled data
X = shuffled_data.drop('species', axis=1)
y = shuffled_data['species']

# Split the shuffled data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize the data
X_train_normalized = normalize_data(X_train)
X_test_normalized = normalize_data(X_test)

# Calculate the covariance matrix of the normalized training data
cov_matrix = calculate_covariance_matrix(X_train_normalized)

# Perform eigen decomposition of the covariance matrix
eigenvalues, eigenvectors = perform_eigen_decomposition(cov_matrix)

# Transform the normalized training and test data using the selected principal components
n_components = 2
X_train_pca = project_data(X_train_normalized, eigenvectors, n_components)
X_test_pca = project_data(X_test_normalized, eigenvectors, n_components)

# Plot histograms of each feature, colored by class label
feature_names = X.columns.tolist()
plot_histograms(X_train_pca, y_train, feature_names)

# Plot scatterplots of the transformed data, colored by class label
plot_scatterplots(X_train_pca, y_train)

# Build the decision tree classifier
tree = DecisionTreeClassifier(max_depth=None)
tree.fit(X_train_pca, y_train)

# Make predictions on the test set
y_pred = tree.predict(X_test_pca)

# Calculate the accuracy of the classifier
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy*100, "%")
