# %%
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
from sklearn.model_selection import cross_val_score, train_test_split,  GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler



# %%
# Load and prepare data
trainDF = pd.read_csv("driving/train_motion_data.csv").dropna()
testDF = pd.read_csv("driving/test_motion_data.csv").dropna()

features = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"]

X_train = trainDF[features]
y_train = trainDF["Class"]
X_test = testDF[features]
y_test = testDF["Class"]

# %%
# Train Gaussian Naive Bayes 
model = GaussianNB()
model.fit(X_train, y_train)

# %%
# Predict and evaluate then data join/split 
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
loss = log_loss(y_test, y_proba)

print(f"Accuracy: {acc:.2f}")
print(f"Classification Report:\n{report}")
print(f"Confusion Matrix:\n{cm}")
print(f"Log Loss: {loss:.2f}")


df = pd.concat([trainDF, testDF], axis=0, join="outer")
X = df[features]
y = np.ravel(df["Class"])

X, X_split, y, y_split = train_test_split(X, y, test_size=0.25, random_state=42)

# %%
# Cross Val Part
newModel = GaussianNB()
modelTune = GridSearchCV(newModel, {}, cv=5, n_jobs=-1)
modelTune.fit(X,y)
cv_model = modelTune.best_estimator_
pred = cv_model.predict(X_split)


# %%
# Metrics
print('accuracy: ', accuracy_score(y_split, pred))
print('precision: ', precision_score(y_split, pred, average=None))
print('recall', recall_score(y_split, pred, average=None))
print('f1', f1_score(y_split, pred, average=None))

pred_proba = modelTune.predict_proba(X_split)
print('loss', log_loss(y_split, pred_proba))