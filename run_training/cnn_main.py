from classifiers.cnn import *
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report



def cnn():
    cnn_model = CNN(vocab_size=7597, embedding_dims=50, filters=100, filter_size=1, hidden_dims=250, batch_size=50, epochs=1)
    cnn_model.preprocess()
    cnn_model.create_embedding_matrix('../glove.6B/glove.6B.50d.txt')
    cnn_model.compile()
    cnn_model.fit()
    cnn_model.predict()
    # evaluation
    print("\nF1 Score: {:.2f}".format(f1_score(cnn_model.y_true, cnn_model.y_pred, average='micro') * 100))
    print("Detailed Report")
    print(classification_report(cnn_model.y_true, cnn_model.y_pred, target_names=cnn_model.class_name))
    # cnn_model.save_model()
    # cnn_model.load_model()
    # return cnn_model

def cnn_with_multi_channels():
    cnn_model = CNN(7597, 50, 100, 1, 250, 50, 30)
    cnn_model.preprocess()
    cnn_model.create_embedding_matrix('../glove.6B/glove.6B.50d.txt')
    cnn_model.define_multi_channels(1, 2, 3)
    cnn_model.compile_multi_channels()
    cnn_model.fit_multi_channels()
    return cnn_model

def load_model(outpath):
    """load saved model and print summary"""
    model = load_model(outpath, compile=True)
    print(model.summary)
    return model


cnn = cnn()
