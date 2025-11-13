
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


def naive_bayes_classifier(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GaussianNB()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    loss = log_loss(y_test, model.predict_proba(X_test_scaled))

    return accuracy, precision, recall, f1, report, confusion, loss


def loadDataset(file_path):
    return pd.read_csv(file_path)


def saveDataset(df, file_path):
    df.to_csv(file_path, index=False)


def load_and_prepare_data():
    """Load and prepare the dataset"""

    # Load datasets
    train_df = pd.read_csv("./driving/train_motion_data.csv")
    test_df = pd.read_csv("./driving/test_motion_data.csv")

    # Prepare features
    features = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"]
    X_train = train_df[features]
    y_train = train_df["Class"]
    X_test = test_df[features]
    y_test = test_df["Class"]

    # Scale features
    return X_train, X_test, y_train, y_test


def trainModel(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def predictModel(model, X_test):
    return model.predict(X_test)


def evaluateModel(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    loss = log_loss(y_test, model.predict_proba(X_test))

    return accuracy, precision, recall, f1, report, confusion, loss


  

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    model = trainModel(X_train, y_train)
    accuracy, precision, recall, f1, report, confusion, loss = evaluateModel(
        model, X_test, y_test
    )
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Classification Report:\n{report}")
    print(f"Confusion Matrix:\n{confusion}")
    print(f"Log Loss: {loss:.2f}")
