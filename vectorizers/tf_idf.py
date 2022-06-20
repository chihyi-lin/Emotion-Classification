# An SVM with rbf kernel without optimization of hyperparameters is used as a classifiers.

# Feature extraction / Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
# Classifier
from sklearn.svm import SVC
# Metrics and plots
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
# Text preprocessing
from preprocessing.preprocessing_baseline import *


def preprocessing(dataset):
    # Create documents as a nested list
    c = Corpus(dataset)
    docs = c.documents
    # Get raw text columns
    X_data = list()
    for i in docs:
        X_data.append(i[1])
    # Tokenizing texts
    d = Document(docs)
    tokenized_data = d.tokenize()
    # Split data into label vectors and text vectors
    tokenized_texts = list()  #[['1st tokenized doc'], ['2nd tokenized doc']]
    y_data = list()
    for i in tokenized_data:
        y_data.append(i[0])
        tokenized_texts.append(i[1])
    return X_data, tokenized_texts, y_data


# convert to feature vector
def dummy_fun(doc):
    return doc

#  TODO: 1.remove too large/low frequency 2.n-gram: 1-3gram, 1+2+3gram
vectorizer = TfidfVectorizer(
    analyzer='word', tokenizer=dummy_fun, preprocessor=dummy_fun, lowercase=False, token_pattern=None)

# Learn vocabulary and idf, return document-term matrix.
X_train, tokenized_texts_train, y_train = preprocessing('data/isear-train.csv')
X = vectorizer.fit_transform(tokenized_texts_train)
# transform testing and training datasets to vectors
X_train_vector = vectorizer.transform(X_train)
X_test, tokenized_texts_test, y_test = preprocessing('data/isear-val.csv')
X_test_vector = vectorizer.transform(X_test)
# feature_name = vectorizer.get_feature_names_out()
# print(feature_name[:50])
# # print(X.toarray())


# Train classifiers
clf = SVC(probability=True, kernel='rbf')
clf.fit(X_train_vector, y_train)
#
predictions = clf.predict(X_test_vector)
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, predictions) * 100))
print("\nF1 Score: {:.2f}".format(f1_score(y_test, predictions, average='micro') * 100))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
#
# class_names = ['joy', 'fear', 'guilt', 'anger', 'shame', 'disgust', 'sadness']
# plot_confusion_matrix(y_test, predictions, classes=class_names, normalize=True, title='Normalized confusion matrix')
# plt.show()

# # predict and evaluate predictions
# predictions = clf.predict_proba(X_test_vector)
# print(predictions[:,1])
# print('ROC-AUC yields ' + str(roc_auc_score(y_test, predictions[:,1])))