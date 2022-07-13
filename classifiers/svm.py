import pandas as pd
import spacy
import string
from spacy.lang.en import English
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
import pickle

nlp = spacy.load('en_core_web_sm')

df_train = pd.read_csv("/Users/yathanlai/Desktop/Pycharm/isear-train.csv", sep='\t', header=None, encoding='utf-8')
df_test = pd.read_csv("/Users/yathanlai/Desktop/Pycharm/isear-test.csv", sep='\t', header=None, encoding='utf-8')
df_train.to_csv('train.csv')
df_train.to_csv('test.csv')

pd.set_option('display.max_column', None)
pd.set_option('display.max_colwidt', 200)
#print(df_train.size)
#print(df_train.columns.tolist()) ##print the name of the column, there is only one
df_train.columns =['Raw'] #rename the only column
df_train = df_train['Raw'].str.split(',', n=1, expand = True)
df_train.columns =['Emotion','Text']

df_test.columns =['Raw']
df_test = df_test['Raw'].str.split(',', n=1, expand = True)
df_test.columns =['Emotion','Text']

print(df_test.shape)
df_test['Text'].dropna(inplace=True)  #check for NaN values and remove them
df_test['Text'] =df_test['Text'].astype(str) #convert the column to string
df_test.columns =['Emotion','Text']
df_test['Text'] = df_test['Text'].str.lower()
df_test['Emotion'] = df_test['Emotion'].str.replace('\n','')
df_test = df_test[df_test['Emotion'].str.contains('joy|disgust|anger|shame|guilt|fear|sadness') == True]
print(df_test['Emotion'].value_counts())

data = pd.concat([df_train, df_test])
print('size of training set: %s' % (len(df_train['Text'])))
print('size of validation set: %s' % (len(df_test['Text'])))

data['Text'].dropna(inplace=True)  #check for NaN values and remove them
data['Text'] = data['Text'].astype(str) #convert the column to string
data.columns =['Emotion','Text']
data['Text'] = data['Text'].str.lower() #set all associated texts to lower case

#remove invalid emotion labels
print(data.shape)
data['Emotion'] = data['Emotion'].str.replace('\n','')
data = data[data['Emotion'].str.contains('joy|disgust|anger|shame|guilt|fear|sadness') == True]
data = data[data['Emotion'].str.contains('felt') == False]
print(data['Emotion'].value_counts())

#plot according to 7 emotion labels
# fig = plt.figure(figsize=(8,6))
# data.groupby('Emotion').Text.count().plot.bar(ylim=0)
# plt.show()

data['Text'] = data['Text'].apply(word_tokenize)
data['Text'] = data['Text'].astype(str)

X_train = df_train.Text.astype(str)
X_test = df_test.Text.astype(str)
y_train = df_train.Emotion.astype(str)
y_test = df_test.Emotion.astype(str)

## Run TF-IDF, the parameter is uni-gram now
vectorizer = TfidfVectorizer(ngram_range=(1,1))
vectorizer.fit_transform(data.Text)
vect_X_train = vectorizer.transform(X_train)
vect_X_test = vectorizer.transform(X_test)

print(vect_X_train.shape)
print(vect_X_test.shape)

classifier = svm.SVC(kernel = 'linear', gamma = 'auto', C=2)
classifier.fit(vect_X_train, y_train)
ysvm_pred = classifier.predict(vect_X_test)

y_test = np.array(y_test)
print(y_test)
y_predict = np.array(ysvm_pred)
print(y_predict)


print("\nAccuracy: {:.2f}%".format(accuracy_score(y_test, ysvm_pred) * 100))
print("\nMacro F1 Score: {:.2f}".format(f1_score(y_test, ysvm_pred, average='macro') * 100))
print("\nMicro F1 Score: {:.2f}".format(f1_score(y_test, ysvm_pred, average='micro') * 100))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, ysvm_pred))

target_names = ['joy', 'disgust', 'anger', 'shame', 'guilt', 'fear', 'sadness']
print(classification_report(y_test, ysvm_pred, target_names=target_names))

