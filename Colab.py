
import pandas as pd


file_path = '/content/Water_Table.csv'

data = pd.read_csv(file_path)


print("First five rows of the dataset:")
print(data.head())


print("\nDataset Information:")
print(data.info())
import matplotlib.pyplot as plt
import seaborn as sns
# Display summary statistics of the dataset
print("\nSummary Statistics of the dataset:")
print(data.describe(include='all'))

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Histograms
data.hist(figsize=(15, 10), bins=20)
plt.suptitle('Histograms of Water Quality Parameters')
plt.show()

# Box Plots
plt.figure(figsize=(15, 10))
sns.boxplot(data)
plt.title('Box Plots of Water Quality Parameters')
plt.xticks(rotation=90)
plt.show()

# Correlation Matrix
plt.figure(figsize=(10, 8))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Water Quality Parameters')
plt.show()
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, balanced_accuracy_score, matthews_corrcoef, precision_score,
                             recall_score, f1_score)

# Prepare features (X) and target (y)
X = data.drop('Potability', axis=1)
y = data['Potability']

# Handle missing values by dropping rows with NaNs in both X and y
X = X.dropna()
y = y.loc[X.index]  # Ensure that the target variable is aligned with the features

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to train KNN and display results
def knn_model_results(X_train_scaled, X_test_scaled, y_train, y_test):
    # Apply K-Nearest Neighbors (KNN) algorithm
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)

    # Predict on test data
    y_pred = knn.predict(X_test_scaled)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Calculate sensitivity, specificity, precision, recall, and f1-score
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)  # Also known as recall
    specificity = tn / (tn + fp)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Classification report
    report = classification_report(y_test, y_pred)

    # Confusion matrix visualization
    ConfusionMatrixDisplay.from_estimator(knn, X_test_scaled, y_test)
    plt.title('Confusion Matrix - KNN Classification')
    plt.show()

    # Bar graph for metric comparison
    metrics = ['Sensitivity', 'Specificity', 'Precision', 'Recall', 'F1-score', 'Support']
    values = [sensitivity, specificity, precision, recall, f1, len(y_test)]

    plt.figure()
    plt.title('K-Nearest Neighbors Classification Metrics Comparison')
    plt.bar(metrics, values)
    plt.ylim(0, 1)
    plt.ylabel('Metric Value')
    plt.show()

    # Return metrics
    return report, accuracy, balanced_accuracy, mcc, sensitivity, specificity

# Function to display the classification report and metrics
def print_results(report, accuracy, balanced_accuracy, mcc, sensitivity, specificity):
    # Print classification report
    print("Classification Report:")
    print(report)

    # Print additional metrics
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print(f"Balanced Accuracy: {balanced_accuracy:.2%}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.2f}")
    print(f"Sensitivity (Recall): {sensitivity:.2%}")
    print(f"Specificity: {specificity:.2%}")

# Call the function to show accuracy and graphs
knn_report, knn_accuracy, knn_balanced_accuracy, knn_mcc, knn_sensitivity, knn_specificity = knn_model_results(X_train_scaled, X_test_scaled, y_train, y_test)
print_results(knn_report, knn_accuracy, knn_balanced_accuracy, knn_mcc, knn_sensitivity, knn_specificity)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, accuracy_score, balanced_accuracy_score,
                             matthews_corrcoef, ConfusionMatrixDisplay, precision_score,
                             recall_score, f1_score, confusion_matrix)
import matplotlib.pyplot as plt

# Function to train Decision Tree and display results
def decision_tree_model(X_train_scaled, X_test_scaled, y_train, y_test):
    # Apply Decision Tree algorithm
    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(X_train_scaled, y_train)

    # Predict on test data
    y_pred = tree.predict(X_test_scaled)

    # Calculate accuracy and other metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Calculate sensitivity, specificity, precision, recall, and f1-score
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)  # Also known as recall
    specificity = tn / (tn + fp)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Classification report
    report = classification_report(y_test, y_pred)

    # Confusion matrix visualization
    ConfusionMatrixDisplay.from_estimator(tree, X_test_scaled, y_test)
    plt.title('Confusion Matrix - Decision Tree Classification')
    plt.show()

    # Bar graph for metric comparison
    metrics = ['Sensitivity', 'Specificity', 'Precision', 'Recall', 'F1-score', 'Support']
    values = [sensitivity, specificity, precision, recall, f1, len(y_test)]

    plt.figure()
    plt.title('Decision Tree Classification Metrics Comparison')
    plt.bar(metrics, values)
    plt.ylim(0, 1)
    plt.ylabel('Metric Value')
    plt.show()

    # Return metrics
    return report, accuracy, balanced_accuracy, mcc, sensitivity, specificity

# Function to display the classification report and additional metrics
def print_results(report, accuracy, balanced_accuracy, mcc, sensitivity, specificity):
    # Print classification report
    print("Classification Report:")
    print(report)

    # Print additional metrics
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print(f"Balanced Accuracy: {balanced_accuracy:.2%}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.2f}")
    print(f"Sensitivity (Recall): {sensitivity:.2%}")
    print(f"Specificity: {specificity:.2%}")

# Call the function to show results for Decision Tree
tree_report, tree_accuracy, tree_balanced_accuracy, tree_mcc, tree_sensitivity, tree_specificity = decision_tree_model(X_train_scaled, X_test_scaled, y_train, y_test)
print_results(tree_report, tree_accuracy, tree_balanced_accuracy, tree_mcc, tree_sensitivity, tree_specificity)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, accuracy_score, balanced_accuracy_score,
                             matthews_corrcoef, ConfusionMatrixDisplay, precision_score,
                             recall_score, f1_score, confusion_matrix)
import matplotlib.pyplot as plt

# Function to train Random Forest and display results
def random_forest_model(X_train_scaled, X_test_scaled, y_train, y_test):
    # Apply Random Forest algorithm
    forest = RandomForestClassifier(random_state=42)
    forest.fit(X_train_scaled, y_train)

    # Predict on test data
    y_pred = forest.predict(X_test_scaled)

    # Calculate accuracy and other metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Calculate sensitivity, specificity, precision, recall, and F1-score
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)  # Also known as recall
    specificity = tn / (tn + fp)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Classification report
    report = classification_report(y_test, y_pred)

    # Confusion matrix visualization
    ConfusionMatrixDisplay.from_estimator(forest, X_test_scaled, y_test)
    plt.title('Confusion Matrix - Random Forest Classification')
    plt.show()

    # Bar graph for metric comparison
    metrics = ['Sensitivity', 'Specificity', 'Precision', 'Recall', 'F1-score', 'Support']
    values = [sensitivity, specificity, precision, recall, f1, len(y_test)]

    plt.figure()
    plt.title('Random Forest Classification Metrics Comparison')
    plt.bar(metrics, values)
    plt.ylim(0, 1)
    plt.ylabel('Metric Value')
    plt.show()

    # Return metrics
    return report, accuracy, balanced_accuracy, mcc, sensitivity, specificity

# Function to display the classification report and additional metrics
def print_results(report, accuracy, balanced_accuracy, mcc, sensitivity, specificity):
    # Print classification report
    print("Classification Report:")
    print(report)

    # Print additional metrics
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print(f"Balanced Accuracy: {balanced_accuracy:.2%}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.2f}")
    print(f"Sensitivity (Recall): {sensitivity:.2%}")
    print(f"Specificity: {specificity:.2%}")

# Call the function to show results for Random Forest
forest_report, forest_accuracy, forest_balanced_accuracy, forest_mcc, forest_sensitivity, forest_specificity = random_forest_model(X_train_scaled, X_test_scaled, y_train, y_test)
print_results(forest_report, forest_accuracy, forest_balanced_accuracy, forest_mcc, forest_sensitivity, forest_specificity)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (classification_report, accuracy_score, balanced_accuracy_score,
                             matthews_corrcoef, ConfusionMatrixDisplay, precision_score,
                             recall_score, f1_score, confusion_matrix)
import matplotlib.pyplot as plt

# Function to train Naive Bayes and display results
def naive_bayes_model(X_train_scaled, X_test_scaled, y_train, y_test):
    # Apply Naive Bayes algorithm
    nb = GaussianNB()
    nb.fit(X_train_scaled, y_train)

    # Predict on test data
    y_pred = nb.predict(X_test_scaled)

    # Calculate accuracy and other metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Calculate sensitivity, specificity, precision, recall, and F1-score
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)  # Also known as recall
    specificity = tn / (tn + fp)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Classification report
    report = classification_report(y_test, y_pred)

    # Confusion matrix visualization
    ConfusionMatrixDisplay.from_estimator(nb, X_test_scaled, y_test)
    plt.title('Confusion Matrix - Naive Bayes Classification')
    plt.show()

    # Bar graph for metric comparison
    metrics = ['Sensitivity', 'Specificity', 'Precision', 'Recall', 'F1-score', 'Support']
    values = [sensitivity, specificity, precision, recall, f1, len(y_test)]

    plt.figure()
    plt.title('Naive Bayes Classification Metrics Comparison')
    plt.bar(metrics, values)
    plt.ylim(0, 1)
    plt.ylabel('Metric Value')
    plt.show()

    # Return metrics
    return report, accuracy, balanced_accuracy, mcc, sensitivity, specificity

# Function to display the classification report and additional metrics
def print_results(report, accuracy, balanced_accuracy, mcc, sensitivity, specificity):
    # Print classification report
    print("Classification Report:")
    print(report)

    # Print additional metrics
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print(f"Balanced Accuracy: {balanced_accuracy:.2%}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.2f}")
    print(f"Sensitivity (Recall): {sensitivity:.2%}")
    print(f"Specificity: {specificity:.2%}")

# Call the function to show results for Naive Bayes
nb_report, nb_accuracy, nb_balanced_accuracy, nb_mcc, nb_sensitivity, nb_specificity = naive_bayes_model(X_train_scaled, X_test_scaled, y_train, y_test)
print_results(nb_report, nb_accuracy, nb_balanced_accuracy, nb_mcc, nb_sensitivity, nb_specificity)
import lightgbm as lgb
from sklearn.metrics import (classification_report, accuracy_score, balanced_accuracy_score,
                             matthews_corrcoef, ConfusionMatrixDisplay, precision_score,
                             recall_score, f1_score, confusion_matrix)
import matplotlib.pyplot as plt

# Function to train LightGBM and display results
def lightgbm_model(X_train_scaled, X_test_scaled, y_train, y_test):
    # Apply LightGBM algorithm
    lgbm = lgb.LGBMClassifier(random_state=42)
    lgbm.fit(X_train_scaled, y_train)

    # Predict on test data
    y_pred = lgbm.predict(X_test_scaled)

    # Calculate accuracy and other metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Calculate sensitivity, specificity, precision, recall, and F1-score
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)  # Also known as recall
    specificity = tn / (tn + fp)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Classification report
    report = classification_report(y_test, y_pred)

    # Confusion matrix visualization
    ConfusionMatrixDisplay.from_estimator(lgbm, X_test_scaled, y_test)
    plt.title('Confusion Matrix - LightGBM Classification')
    plt.show()

    # Bar graph for metric comparison
    metrics = ['Sensitivity', 'Specificity', 'Precision', 'Recall', 'F1-score', 'Support']
    values = [sensitivity, specificity, precision, recall, f1, len(y_test)]

    plt.figure()
    plt.title('LightGBM Classification Metrics Comparison')
    plt.bar(metrics, values)
    plt.ylim(0, 1)
    plt.ylabel('Metric Value')
    plt.show()

    # Return metrics
    return report, accuracy, balanced_accuracy, mcc, sensitivity, specificity

# Function to display the classification report and additional metrics
def print_results(report, accuracy, balanced_accuracy, mcc, sensitivity, specificity):
    # Print classification report
    print("Classification Report:")
    print(report)

    # Print additional metrics
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print(f"Balanced Accuracy: {balanced_accuracy:.2%}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.2f}")
    print(f"Sensitivity (Recall): {sensitivity:.2%}")
    print(f"Specificity: {specificity:.2%}")

# Call the function to show results for LightGBM
lgbm_report, lgbm_accuracy, lgbm_balanced_accuracy, lgbm_mcc, lgbm_sensitivity, lgbm_specificity = lightgbm_model(X_train_scaled, X_test_scaled, y_train, y_test)
print_results(lgbm_report, lgbm_accuracy, lgbm_balanced_accuracy, lgbm_mcc, lgbm_sensitivity, lgbm_specificity)
import xgboost as xgb
from sklearn.metrics import (classification_report, accuracy_score, balanced_accuracy_score,
                             matthews_corrcoef, ConfusionMatrixDisplay, precision_score,
                             recall_score, f1_score, confusion_matrix)
import matplotlib.pyplot as plt

# Function to train XGBoost and display results
def xgboost_model(X_train_scaled, X_test_scaled, y_train, y_test):
    # Apply XGBoost algorithm
    xgboost = xgb.XGBClassifier(random_state=42)
    xgboost.fit(X_train_scaled, y_train)

    # Predict on test data
    y_pred = xgboost.predict(X_test_scaled)

    # Calculate accuracy and other metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Calculate sensitivity, specificity, precision, recall, and F1-score
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)  # Also known as recall
    specificity = tn / (tn + fp)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Classification report
    report = classification_report(y_test, y_pred)

    # Confusion matrix visualization
    ConfusionMatrixDisplay.from_estimator(xgboost, X_test_scaled, y_test)
    plt.title('Confusion Matrix - XGBoost Classification')
    plt.show()

    # Bar graph for metric comparison
    metrics = ['Sensitivity', 'Specificity', 'Precision', 'Recall', 'F1-score', 'Support']
    values = [sensitivity, specificity, precision, recall, f1, len(y_test)]

    plt.figure()
    plt.title('XGBoost Classification Metrics Comparison')
    plt.bar(metrics, values)
    plt.ylim(0, 1)
    plt.ylabel('Metric Value')
    plt.show()

    # Return metrics
    return report, accuracy, balanced_accuracy, mcc, sensitivity, specificity

# Function to display the classification report and additional metrics
def print_results(report, accuracy, balanced_accuracy, mcc, sensitivity, specificity):
    # Print classification report
    print("Classification Report:")
    print(report)

    # Print additional metrics
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print(f"Balanced Accuracy: {balanced_accuracy:.2%}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.2f}")
    print(f"Sensitivity (Recall): {sensitivity:.2%}")
    print(f"Specificity: {specificity:.2%}")

# Call the function to show results for XGBoost
xgb_report, xgb_accuracy, xgb_balanced_accuracy, xgb_mcc, xgb_sensitivity, xgb_specificity = xgboost_model(X_train_scaled, X_test_scaled, y_train, y_test)
print_results(xgb_report, xgb_accuracy, xgb_balanced_accuracy, xgb_mcc, xgb_sensitivity, xgb_specificity)
import catboost as cb
from sklearn.metrics import (classification_report, accuracy_score, balanced_accuracy_score,
                             matthews_corrcoef, ConfusionMatrixDisplay, precision_score,
                             recall_score, f1_score, confusion_matrix)
import matplotlib.pyplot as plt

# Function to train CatBoost and display results
def catboost_model(X_train_scaled, X_test_scaled, y_train, y_test):
    # Apply CatBoost algorithm
    catboost = cb.CatBoostClassifier(random_state=42, verbose=0)  # Set verbose=0 to suppress detailed output
    catboost.fit(X_train_scaled, y_train)

    # Predict on test data
    y_pred = catboost.predict(X_test_scaled)

    # Calculate accuracy and other metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Calculate sensitivity, specificity, precision, recall, and F1-score
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)  # Also known as recall
    specificity = tn / (tn + fp)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Classification report
    report = classification_report(y_test, y_pred)

    # Confusion matrix visualization
    ConfusionMatrixDisplay.from_estimator(catboost, X_test_scaled, y_test)
    plt.title('Confusion Matrix - CatBoost Classification')
    plt.show()

    # Bar graph for metric comparison
    metrics = ['Sensitivity', 'Specificity', 'Precision', 'Recall', 'F1-score', 'Support']
    values = [sensitivity, specificity, precision, recall, f1, len(y_test)]

    plt.figure()
    plt.title('CatBoost Classification Metrics Comparison')
    plt.bar(metrics, values)
    plt.ylim(0, 1)
    plt.ylabel('Metric Value')
    plt.show()

    # Return metrics
    return report, accuracy, balanced_accuracy, mcc, sensitivity, specificity

# Function to display the classification report and additional metrics
def print_results(report, accuracy, balanced_accuracy, mcc, sensitivity, specificity):
    # Print classification report
    print("Classification Report:")
    print(report)

    # Print additional metrics
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print(f"Balanced Accuracy: {balanced_accuracy:.2%}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.2f}")
    print(f"Sensitivity (Recall): {sensitivity:.2%}")
    print(f"Specificity: {specificity:.2%}")

# Call the function to show results for CatBoost
cat_report, cat_accuracy, cat_balanced_accuracy, cat_mcc, cat_sensitivity, cat_specificity = catboost_model(X_train_scaled, X_test_scaled, y_train, y_test)
print_results(cat_report, cat_accuracy, cat_balanced_accuracy, cat_mcc, cat_sensitivity, cat_specificity)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (classification_report, accuracy_score, balanced_accuracy_score,
                             matthews_corrcoef, ConfusionMatrixDisplay, precision_score,
                             recall_score, f1_score, confusion_matrix)
import matplotlib.pyplot as plt

# Function to train AdaBoost and display results
def adaboost_model(X_train_scaled, X_test_scaled, y_train, y_test):
    # Apply AdaBoost algorithm
    adaboost = AdaBoostClassifier(random_state=42)
    adaboost.fit(X_train_scaled, y_train)

    # Predict on test data
    y_pred = adaboost.predict(X_test_scaled)

    # Calculate accuracy and other metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Calculate sensitivity, specificity, precision, recall, and F1-score
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)  # Also known as recall
    specificity = tn / (tn + fp)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Classification report
    report = classification_report(y_test, y_pred)

    # Confusion matrix visualization
    ConfusionMatrixDisplay.from_estimator(adaboost, X_test_scaled, y_test)
    plt.title('Confusion Matrix - AdaBoost Classification')
    plt.show()

    # Bar graph for metric comparison
    metrics = ['Sensitivity', 'Specificity', 'Precision', 'Recall', 'F1-score', 'Support']
    values = [sensitivity, specificity, precision, recall, f1, len(y_test)]

    plt.figure()
    plt.title('AdaBoost Classification Metrics Comparison')
    plt.bar(metrics, values)
    plt.ylim(0, 1)
    plt.ylabel('Metric Value')
    plt.show()

    # Return metrics
    return report, accuracy, balanced_accuracy, mcc, sensitivity, specificity

# Function to display the classification report and additional metrics
def print_results(report, accuracy, balanced_accuracy, mcc, sensitivity, specificity):
    # Print classification report
    print("Classification Report:")
    print(report)

    # Print additional metrics
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print(f"Balanced Accuracy: {balanced_accuracy:.2%}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.2f}")
    print(f"Sensitivity (Recall): {sensitivity:.2%}")
    print(f"Specificity: {specificity:.2%}")

# Call the function to show results for AdaBoost
ada_report, ada_accuracy, ada_balanced_accuracy, ada_mcc, ada_sensitivity, ada_specificity = adaboost_model(X_train_scaled, X_test_scaled, y_train, y_test)
print_results(ada_report, ada_accuracy, ada_balanced_accuracy, ada_mcc, ada_sensitivity, ada_specificity)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, accuracy_score, balanced_accuracy_score,
                             matthews_corrcoef, ConfusionMatrixDisplay, precision_score,
                             recall_score, f1_score, confusion_matrix)
import matplotlib.pyplot as plt

# Function to train Neural Network and display results
def neural_network_model(X_train_scaled, X_test_scaled, y_train, y_test):
    # Apply Neural Network (MLPClassifier) algorithm
    nn_model = MLPClassifier(random_state=42, max_iter=1000)  # max_iter is set higher to ensure convergence
    nn_model.fit(X_train_scaled, y_train)

    # Predict on test data
    y_pred = nn_model.predict(X_test_scaled)

    # Calculate accuracy and other metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Calculate sensitivity, specificity, precision, recall, and F1-score
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)  # Also known as recall
    specificity = tn / (tn + fp)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Classification report
    report = classification_report(y_test, y_pred)

    # Confusion matrix visualization
    ConfusionMatrixDisplay.from_estimator(nn_model, X_test_scaled, y_test)
    plt.title('Confusion Matrix - Neural Network Classification')
    plt.show()

    # Bar graph for metric comparison
    metrics = ['Sensitivity', 'Specificity', 'Precision', 'Recall', 'F1-score', 'Support']
    values = [sensitivity, specificity, precision, recall, f1, len(y_test)]

    plt.figure()
    plt.title('Neural Network Classification Metrics Comparison')
    plt.bar(metrics, values)
    plt.ylim(0, 1)
    plt.ylabel('Metric Value')
    plt.show()

    # Return metrics
    return report, accuracy, balanced_accuracy, mcc, sensitivity, specificity

# Function to display the classification report and additional metrics
def print_results(report, accuracy, balanced_accuracy, mcc, sensitivity, specificity):
    # Print classification report
    print("Classification Report:")
    print(report)

    # Print additional metrics
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print(f"Balanced Accuracy: {balanced_accuracy:.2%}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.2f}")
    print(f"Sensitivity (Recall): {sensitivity:.2%}")
    print(f"Specificity: {specificity:.2%}")

# Call the function to show results for Neural Network
nn_report, nn_accuracy, nn_balanced_accuracy, nn_mcc, nn_sensitivity, nn_specificity = neural_network_model(X_train_scaled, X_test_scaled, y_train, y_test)
print_results(nn_report, nn_accuracy, nn_balanced_accuracy, nn_mcc, nn_sensitivity, nn_specificity)
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, accuracy_score, balanced_accuracy_score,
                             matthews_corrcoef, ConfusionMatrixDisplay, precision_score,
                             recall_score, f1_score, confusion_matrix)
import matplotlib.pyplot as plt

# Function to train SVM and display results
def svm_model(X_train_scaled, X_test_scaled, y_train, y_test):
    # Apply SVM algorithm
    svm_model = SVC(random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    # Predict on test data
    y_pred = svm_model.predict(X_test_scaled)

    # Calculate accuracy and other metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Calculate sensitivity, specificity, precision, recall, and F1-score
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)  # Also known as recall
    specificity = tn / (tn + fp)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Classification report
    report = classification_report(y_test, y_pred)

    # Confusion matrix visualization
    ConfusionMatrixDisplay.from_estimator(svm_model, X_test_scaled, y_test)
    plt.title('Confusion Matrix - SVM Classification')
    plt.show()

    # Bar graph for metric comparison
    metrics = ['Sensitivity', 'Specificity', 'Precision', 'Recall', 'F1-score', 'Support']
    values = [sensitivity, specificity, precision, recall, f1, len(y_test)]

    plt.figure()
    plt.title('SVM Classification Metrics Comparison')
    plt.bar(metrics, values)
    plt.ylim(0, 1)
    plt.ylabel('Metric Value')
    plt.show()

    # Return metrics
    return report, accuracy, balanced_accuracy, mcc, sensitivity, specificity

# Function to display the classification report and additional metrics
def print_results(report, accuracy, balanced_accuracy, mcc, sensitivity, specificity):
    # Print classification report
    print("Classification Report:")
    print(report)

    # Print additional metrics
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    print(f"Balanced Accuracy: {balanced_accuracy:.2%}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.2f}")
    print(f"Sensitivity (Recall): {sensitivity:.2%}")
    print(f"Specificity: {specificity:.2%}")

# Call the function to show results for SVM
svm_report, svm_accuracy, svm_balanced_accuracy, svm_mcc, svm_sensitivity, svm_specificity = svm_model(X_train_scaled, X_test_scaled, y_train, y_test)
print_results(svm_report, svm_accuracy, svm_balanced_accuracy, svm_mcc, svm_sensitivity, svm_specificity)
import matplotlib.pyplot as plt
import numpy as np

# Replace these accuracies with your actual results
accuracy_scores = {
    "Decision Tree": 61.09,
    "Random Forest": 67.72,
    "SVM": 68.05,
    "KNN": 62.91,
    "Naive Bayes": 60.60,
    "LightGBM" :63.91,
    "XGBoost": 66.39,
    "CatBoost": 67.88,
    "Neural Network": 67.55,
    "AdaBoost": 58.94,
}

# Sort the accuracy scores by value
sorted_accuracies = dict(sorted(accuracy_scores.items(), key=lambda item: item[1], reverse=True))

# Extract model names and accuracy values
models = list(sorted_accuracies.keys())
accuracies = list(sorted_accuracies.values())

# Plotting the comparison chart
plt.figure(figsize=(10, 6))
bars = plt.barh(models, accuracies, color='skyblue')

# Annotating the bars with their accuracy values
for bar in bars:
    plt.text(bar.get_width() - 5, bar.get_y() + bar.get_height()/2,
             f'{bar.get_width():.2f}%', va='center', ha='center', color='white', weight='bold')

# Title and labels
plt.title("Comparison of Model Accuracies", fontsize=14)
plt.xlabel("Accuracy (%)", fontsize=12)
plt.ylabel("Models", fontsize=12)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt

# Models
models = ['Decision Tree', 'Random Forest', 'SVM', 'KNN', 'Naive Bayes', 'LightGBM', 'XGBoost', 'CatBoost', 'Neural Network', 'AdaBoost']

# Metrics values
accuracy = [0.6109, 0.6772, 0.6805, 0.6291, 0.6060, 0.6391, 0.6639, 0.6788, 0.6755, 0.5894]
specificity = [0.60, 0.70, 0.65, 0.63, 0.58, 0.64, 0.68, 0.69, 0.66, 0.55]
sensitivity = [0.62, 0.66, 0.71, 0.63, 0.63, 0.64, 0.65, 0.67, 0.69, 0.63]
f1_score = [0.61, 0.66, 0.68, 0.63, 0.60, 0.64, 0.66, 0.68, 0.67, 0.59]
balanced_accuracy = [0.61, 0.68, 0.68, 0.63, 0.60, 0.64, 0.66, 0.68, 0.67, 0.59]
mcc = [0.22, 0.35, 0.36, 0.26, 0.21, 0.28, 0.33, 0.36, 0.34, 0.18]

# Dictionary for metrics data
metrics = {
    'Accuracy': accuracy,
    'Specificity': specificity,
    'Sensitivity': sensitivity,
    'F1-score': f1_score,
    'Balanced Accuracy': balanced_accuracy,
    'MCC': mcc
}

# Bar plot setup
x = range(len(models))
width = 0.12  # Width of each bar

fig, ax = plt.subplots(figsize=(14, 6))  # Set figure size

# Plot each metric as a separate bar within each group
for i, (metric_name, metric_values) in enumerate(metrics.items()):
    ax.bar([xi + i * width for xi in x], metric_values, width, label=metric_name)

# Set labels, title, and legend
ax.set_xlabel('Models', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Comparison of Model Metrics', fontsize=14)
ax.set_xticks([xi + 2.5 * width for xi in x])  # Center x-ticks
ax.set_xticklabels(models, rotation=45, ha='right')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside plot

plt.ylim(0, 1)  # Set y-axis scale to 1
plt.tight_layout()  # Adjust layout to fit all labels and legend
plt.show()
from pyngrok import ngrok
ngrok.set_auth_token("2o1PVSrtWV35p8mAYGNj5unkwL6_3zbEqpUv37ehhc5bV9F8H")
# Decision Tree
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train_scaled, y_train)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(random_state=42)
forest.fit(X_train_scaled, y_train)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)

# LightGBM
import lightgbm as lgb
lgbm = lgb.LGBMClassifier(random_state=42)
lgbm.fit(X_train_scaled, y_train)

# XGBoost
import xgboost as xgb
xgboost = xgb.XGBClassifier(random_state=42)
xgboost.fit(X_train_scaled, y_train)

# CatBoost
import catboost as cb
catboost = cb.CatBoostClassifier(random_state=42, verbose=0)
catboost.fit(X_train_scaled, y_train)

# AdaBoost
from sklearn.ensemble import AdaBoostClassifier
adaboost = AdaBoostClassifier(random_state=42)
adaboost.fit(X_train_scaled, y_train)

# Neural Network
from sklearn.neural_network import MLPClassifier
nn_model = MLPClassifier(random_state=42, max_iter=1000)
nn_model.fit(X_train_scaled, y_train)

# Support Vector Machine (SVM)
from sklearn.svm import SVC
svm_model = SVC(random_state=42)
svm_model.fit(X_train_scaled, y_train)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Load your data (replace this with your actual data loading code)
# data = pd.read_csv('your_data.csv')

# Prepare features and target
X = data.drop('Potability', axis=1)
y = data['Potability']

# Handle missing values
X = X.dropna()
y = y.loc[X.index]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Train and save each model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train_scaled, y_train)
with open('decision_tree_model.pkl', 'wb') as f:
    pickle.dump(tree, f)

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(random_state=42)
forest.fit(X_train_scaled, y_train)
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(forest, f)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
with open('naive_bayes_model.pkl', 'wb') as f:
    pickle.dump(nb, f)

import lightgbm as lgb
lgbm = lgb.LGBMClassifier(random_state=42)
lgbm.fit(X_train_scaled, y_train)
with open('lightgbm_model.pkl', 'wb') as f:
    pickle.dump(lgbm, f)

import xgboost as xgb
xgboost = xgb.XGBClassifier(random_state=42)
xgboost.fit(X_train_scaled, y_train)
with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgboost, f)

import catboost as cb
catboost = cb.CatBoostClassifier(random_state=42, verbose=0)
catboost.fit(X_train_scaled, y_train)
with open('catboost_model.pkl', 'wb') as f:
    pickle.dump(catboost, f)

from sklearn.ensemble import AdaBoostClassifier
adaboost = AdaBoostClassifier(random_state=42)
adaboost.fit(X_train_scaled, y_train)
with open('adaboost_model.pkl', 'wb') as f:
    pickle.dump(adaboost, f)

from sklearn.neural_network import MLPClassifier
nn_model = MLPClassifier(random_state=42, max_iter=1000)
nn_model.fit(X_train_scaled, y_train)
with open('neural_network_model.pkl', 'wb') as f:
    pickle.dump(nn_model, f)

from sklearn.svm import SVC
svm_model = SVC(random_state=42)
svm_model.fit(X_train_scaled, y_train)
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)
%%writefile app.py
import streamlit as st
import numpy as np
import pickle

# Load the pre-trained scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load all pre-trained models
with open('knn_model.pkl', 'rb') as f:
    knn = pickle.load(f)

with open('decision_tree_model.pkl', 'rb') as f:
    decision_tree = pickle.load(f)

with open('random_forest_model.pkl', 'rb') as f:
    random_forest = pickle.load(f)

with open('naive_bayes_model.pkl', 'rb') as f:
    naive_bayes = pickle.load(f)

with open('lightgbm_model.pkl', 'rb') as f:
    lightgbm = pickle.load(f)

with open('xgboost_model.pkl', 'rb') as f:
    xgboost = pickle.load(f)

with open('catboost_model.pkl', 'rb') as f:
    catboost = pickle.load(f)

with open('adaboost_model.pkl', 'rb') as f:
    adaboost = pickle.load(f)

with open('neural_network_model.pkl', 'rb') as f:
    neural_network = pickle.load(f)

with open('svm_model.pkl', 'rb') as f:
    svm = pickle.load(f)

# Map the model names to the model objects
model_functions = {
    "K-Nearest Neighbors": knn,
    "Decision Tree": decision_tree,
    "Random Forest": random_forest,
    "Naive Bayes": naive_bayes,
    "LightGBM": lightgbm,
    "XGBoost": xgboost,
    "CatBoost": catboost,
    "AdaBoost": adaboost,
    "Neural Network": neural_network,
    "Support Vector Machine": svm,
}

# Streamlit App Interface
st.title("Ground Water Quality Measure using Advanced Machine Learning Techniques")

# Team Section
st.markdown(" Team Members:")
st.write("""
- RA2211003011436 YETUKURI GARGEYA SREENADH
- RA2211003011437 KODUMURU ASHISH
- RA2211003011421 SARABU SUHAS
- RA2211003011416 VURUKONDA SAI TARUN
""")

# Step 1: Model Selection
st.sidebar.header("Choose an Algorithm")
model_name = st.sidebar.selectbox("Algorithm", list(model_functions.keys()))
model = model_functions[model_name]

# Step 2: Input for Prediction
st.header(f"{model_name} Prediction Page")
st.write("Enter water quality attributes to predict potability:")

# Input fields for user to enter data
ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
hardness = st.number_input("Hardness", min_value=0.0, value=150.0)
solids = st.number_input("Solids", min_value=0.0, value=20000.0)
chloramines = st.number_input("Chloramines", min_value=0.0, value=7.0)
sulfate = st.number_input("Sulfate", min_value=0.0, value=300.0)
conductivity = st.number_input("Conductivity", min_value=0.0, value=300.0)
organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=20.0)
trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, value=70.0)
turbidity = st.number_input("Turbidity", min_value=0.0, value=4.0)

# Collect inputs into an array
input_data = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])

# Scale the input data using the loaded scaler
scaled_data = scaler.transform(input_data)

# Prediction
if st.button("Predict"):
    prediction = model.predict(scaled_data)
    result = "Potable" if prediction == 1 else "Not Potable"
    st.write(f"The water is predicted to be: **{result}**")


from pyngrok import ngrok
import subprocess

# Start Streamlit app on port 8501
subprocess.Popen(['streamlit', 'run', 'app.py', '--server.port', '8501'])

# Set up ngrok tunnel with explicit HTTP version
public_url = ngrok.connect(8501, bind_tls=True)
print("Public URL:", public_url)
from pyngrok import ngrok

# Disconnect all ngrok tunnels
ngrok.kill()

# Start a fresh ngrok tunnel
public_url = ngrok.connect(5000)
print("Public URL:", public_url)
