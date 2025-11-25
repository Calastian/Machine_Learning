# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import KBinsDiscretizer

from sklearn.model_selection import train_test_split, validation_curve, GridSearchCV

from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.metrics import f1_score as f1
from sklearn.metrics import log_loss

# %% [markdown]
# #SET RANDOM STATE

# %%
RANDOM_STATE = 42

# %% [markdown]
# #READ DATA

# %%
trainDF = pd.read_csv("../driving/train_motion_data.csv").dropna()
testDF = pd.read_csv("../driving/test_motion_data.csv").dropna()

# %% [markdown]
# #NO CROSS 

# %%
def getWindowSampleForColumn(columnChunk:pd.Series)->dict[str, float]:
    chunk = columnChunk.to_numpy()
    
    mag_chunk = np.abs(chunk)
    
    max = np.max(mag_chunk)
    
    min = np.min(mag_chunk)
    
    median = np.median(mag_chunk)
    
    mean = np.average(mag_chunk)
    
    standard_dev = np.std(mag_chunk)
    
    #z score (data - mean)/std
    z_scores = (mag_chunk - standard_dev)/mean
    max_z_score = np.max(z_scores)
    min_z_score = np.min(z_scores)
    
    diffs = np.diff(mag_chunk)
    max_difference = np.max(diffs)
    min_difference = np.min(diffs)
    
    tmp = {'max':max, 'min':min, 'median':median, 'mean':mean, 'median': median,
           'standard_dev': standard_dev, 'max_z_score':max_z_score, 'min_z_score':min_z_score,
           'max_difference': max_difference, 'min_difference': min_difference}
    return {str(columnChunk.name) + '_' + k : tmp[k] for k in tmp}
    
#%%
def makeWindowedData(df:pd.DataFrame, windowSize)->pd.DataFrame:
    df = df.sort_values("Timestamp").reset_index(drop=True)
    
    cols = df.columns.drop(['Timestamp', 'Class'])
    
    tmpDF = pd.DataFrame()
    
    for i in range(len(df) - windowSize + 1):
        window = df.iloc[i:i+windowSize]
        
        row_stats = {}
        
        for col in cols:
            col_stats = getWindowSampleForColumn(window[col])
            row_stats.update(col_stats)
            
        class_ = None
        if 'AGGRESSIVE' in window['Class']:
            class_ = 'AGGRESSIVE'
        elif 'NORMAL' in window['Class']:
            class_ = 'NORMAL'
        else:
            class_ = 'SLOW'
        row_stats.update({'Class':class_})
            
        tmpDF.iloc[i] = row_stats
     
    return tmpDF

# %%
trainDF = makeWindowedData(trainDF, 10)
testDF = makeWindowedData(testDF, 10) 

# %%
# Select input and output features
X = trainDF.drop(columns=['Class'])
y = np.ravel(trainDF[['Class']])

# %%
# Initialize and fit random forest classifier
rfModel = RandomForestClassifier(random_state=RANDOM_STATE)
rfModel.fit(X, y)

# %%
# Calculate prediction accuracy
testX = testDF.drop(columns=['Class'])
testy = np.ravel(testDF[['Class']])

ConfusionMatrixDisplay.from_estimator(rfModel, testX, testy)

# %%
pred = rfModel.predict(testX)

print('accuracy: ', accuracy(testy, pred))
print('precision: ', precision(testy, pred, average=None))
print('recall', recall(testy, pred, average=None))
print('f1', f1(testy, pred, average=None))

pred_proba = rfModel.predict_proba(testX)
print('loss', log_loss(testy, pred_proba))

print('len: ', len(testy))
print('pred: ', pred)
print('true: ', testy)

# %%
