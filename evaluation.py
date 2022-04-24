import re
import pandas as pd

# Load Dataset, add header, start index from 1
df = pd.read_csv("isear-train.csv", names=["Emotion", "Text"], on_bad_lines='skip')
df.index = df.index+1
print(df.head())

# Value counts
emotion_counts = df['Emotion'].value_counts()
print(f'original counts: {emotion_counts}')
print(f'shape: {df.shape}')
print(f'describe: {df.describe()}')

# drop invalid rows
df = df.loc[df['Emotion'].str.contains('joy|guilt|sadness|anger|shame|disgust|fear', regex=True)]
n_emotion_counts = df['Emotion'].value_counts()
print(f'new counts: {n_emotion_counts}')
print(f'new shape: {df.shape}')
print(f'new describe: {df.describe()}')

# Add new columns [Actual joy] and [Predicted joy]
df['Actual joy'] = 0
df.loc[df['Emotion'] == 'joy', 'Actual joy'] = 1
print(df.head(5))
"""
4 invalid rows still remain:
Need to be removed: 1
Need to be categorized into data frame: 3
"""

# ? Predict all texts as joy and calculate the f-score?

# F-score formula
# f_score = 2 * (precision * recall) / (precision + recall)
# precision = tp / (tp+fp)
# recall = tp / (tp+fn)
"""
Emotion / Text / Actual joy / Predicted joy
                  0/1            0/1

Actual Classes:
joy - 1: 777
not joy - 0: 5333-777=4556

Prediction:
joy - 1: 5333
not joy: 0

Confusion matrix:
         Actual class
model    tp:777     fp:4556
         fn:0       tn:0

precision: 777/4556 = 0.1705
recall: 777/777 = 1
f1-score = 2*0.17*1 / 0.17+1 = 0.34/1.17 = 0.29

"""
def compute_tp_tn_fn_fp():
    tp = sum(act == 1) & (pred == 1)