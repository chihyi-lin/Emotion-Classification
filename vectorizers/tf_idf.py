# An SVM with rbf kernel without optimization of hyperparameters is used as a classifiers.

from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing.preprocessing_baseline import *


def preprocessing(dataset):
    # Create documents as a nested list, tokenizing texts
    c = Corpus(dataset)
    docs = c.documents
    d = Document(docs)
    tokenized_data = d.tokenize()

    # Split data into label vector and text vector
    tokenized_docs = list()  #[['1st doc'], ['2nd doc']]
    labels = list()
    for i in tokenized_data:
        labels.append(i[0])
        tokenized_docs.append(i[1])
    return tokenized_docs, labels

# convert to feature vector
def dummy_fun(doc):
    return doc

vectorizer = TfidfVectorizer(
    analyzer='word', tokenizer=dummy_fun, preprocessor=dummy_fun, lowercase=False, token_pattern=None)

# Learn vocabulary and idf, return document-term matrix.
preprocessing('isear-train.csv')
X = vectorizer.fit_transform(tokenized_docs)
# feature_name = vectorizers.get_feature_names_out()
# print(feature_name)
# print(X.toarray())

# # split into training- and test set
# TRAINING_END = date(2014,12,31)
# num_training = len(data[pandas.to_datetime(data["Date"]) <= TRAINING_END])
X_train = X
X_test =
y_train = labels
y_test =

# # train classifiers
# clf = SVC(probability=True, kernel='rbf')
# clf.fit(X_train, y_train)
#
# # predict and evaluate predictions
# predictions = clf.predict_proba(X_test)
# print('ROC-AUC yields ' + str(roc_auc_score(y_test, predictions[:,1])))