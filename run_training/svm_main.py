'''Author: Yat Han Lai'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import re
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion


df_train = pd.read_csv("/Users/yathanlai/Desktop/Pycharm/isear-train.csv", sep='\t', header=None, encoding='utf-8')
df_test = pd.read_csv("/Users/yathanlai/Desktop/Pycharm/isear-test.csv", sep='\t', header=None, encoding='utf-8')
df_val = pd.read_csv("/Users/yathanlai/Desktop/Pycharm/isear-val.csv", sep='\t', header=None, encoding='utf-8')

# pd.set_option('display.max_column', None)
# pd.set_option('display.max_colwidt', 200)

'''data cleaning by renaming the columns and removing invalid data'''
df_train.columns =['Raw']
df_train = df_train['Raw'].str.split(',', n=1, expand = True)
df_train.columns =['Emotion','Text']
df_test.columns =['Raw']
df_test = df_test['Raw'].str.split(',', n=1, expand = True)
df_test.columns =['Emotion','Text']
df_val.columns =['Raw']
df_val = df_val['Raw'].str.split(',', n=1, expand = True)
df_val.columns =['Emotion','Text']
df_val['Text'].dropna(inplace=True)  #check for NaN values and remove them
df_val['Text'] =df_val['Text'].astype(str) #convert the column to string
df_val.columns =['Emotion','Text']
df_val['Text'] = df_test['Text'].str.lower()
df_val['Emotion'] = df_test['Emotion'].str.replace('\n','')
df_val = df_val[df_val['Emotion'].str.contains('joy|disgust|anger|shame|guilt|fear|sadness') == True]
df_test['Text'].dropna(inplace=True)  #check for NaN values and remove them
df_test['Text'] =df_test['Text'].astype(str) #convert the column to string
df_test.columns =['Emotion','Text']
df_test['Text'] = df_test['Text'].str.lower()
df_test['Emotion'] = df_test['Emotion'].str.replace('\n','')
df_test = df_test[df_test['Emotion'].str.contains('joy|disgust|anger|shame|guilt|fear|sadness') == True]
data = pd.concat([df_train, df_test])
data['Text'].dropna(inplace=True)  #check for NaN values and remove them
data['Text'] = data['Text'].astype(str) #convert the column to string
data.columns =['Emotion','Text']
data['Text'] = data['Text'].str.lower() #set all associated texts to lower case
data['Emotion'] = data['Emotion'].str.replace('\n','')
data = data[data['Emotion'].str.contains('joy|disgust|anger|shame|guilt|fear|sadness') == True]
data = data[data['Emotion'].str.contains('felt') == False]

'''Preprocessing'''
'''uncomment to remove puntuation'''
# def remove_punctuation(data):
#     data_nopunt = "".join([c for c in data if c not in string.punctuation])
#     return data_nopunt
# data['Text']=data['Text'].apply(lambda x: remove_punctuation(x))
#
'''tokenization with NLTK'''
data['Text'] = data['Text'].apply(word_tokenize)
#
'''uncomment to remove stop words'''
# stopwords = nltk.corpus.stopwords.words('english')
# def remove_stopwords(data):
#     data_clean = [word for word in data if word not in stopwords]
#     return data_clean
# data['Text']=data['Text'].apply(lambda x:remove_stopwords(x))

data['Text'] = data['Text'].astype(str)


'''uncomment to plot the histogram of 7 emotion labels'''
# fig = plt.figure(figsize=(8,6))
# data.groupby('Emotion').Text.count().plot.bar(ylim=0)
# plt.show()


X_train = df_train.Text.astype(str)
X_test = df_test.Text.astype(str)
y_train = df_train.Emotion.astype(str)
y_test = df_test.Emotion.astype(str) #true labels

'''Run TF-IDF: n-gram is set to unigram here'''
vectorizer = TfidfVectorizer(ngram_range=(1,1))
'uncomment to combine unigrams, bigrams and trigrams'
# vectorizer1 = TfidfVectorizer(ngram_range=(1,2))
# vectorizer2= TfidfVectorizer(ngram_range=(2,3))
# vectorizer = FeatureUnion([('vectorizer1', vectorizer1), ('vectorizer2', vectorizer2)])
vectorizer.fit_transform(data.Text) #feed the whole corpus and learn the vocab
vect_X_train = vectorizer.transform(X_train)  #no. of features according to n-grams
vect_X_test = vectorizer.transform(X_test)

'''classifier'''
classifier = svm.SVC(kernel='linear', C=2)
classifier.fit(vect_X_train, y_train)
ysvm_pred = classifier.predict(vect_X_test)
y_test = np.array(y_test)
ysvm_pred = np.array(ysvm_pred)

'''Evaluation'''
#print(classifier.score(vect_X_train, y_train)) #compare the seen in df_train: ca. 98%
#print(classifier.score(vect_X_test, y_test)) #compare labels and text in df_test ca. 55%

print("\nAccuracy: {:.2f}%".format(accuracy_score(y_test, ysvm_pred) * 100))
print("\nMacro F1 Score: {:.2f}".format(f1_score(y_test, ysvm_pred, average='macro') * 100))
print("\nMicro F1 Score: {:.2f}".format(f1_score(y_test, ysvm_pred, average='micro') * 100))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, ysvm_pred))

target_names = ['joy', 'disgust', 'anger', 'shame', 'guilt', 'fear', 'sadness']
print(classification_report(y_test, ysvm_pred, target_names=target_names))




'''uncomment to show misclassified instances'''
# for row_index, (input, prediction, label) in enumerate(zip (X_test, ysvm_pred, y_test)):
#   if prediction != label:
#     print('Row', row_index, 'has been classified as ', prediction, 'and should be ', label)
# X_test.to_csv('xtest.csv')


'''confusion matrix'''
class_names = ['anger','disgust','fear','guilt','joy', 'sadness','shame']
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    '''
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()

    # Set size
    fig.set_size_inches(13, 7.5)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.grid(False)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

plot_confusion_matrix(y_test, ysvm_pred, classes=class_names, normalize=True, title='Normalized confusion matrix')
plt.show()

