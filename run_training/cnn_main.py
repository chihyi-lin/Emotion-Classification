"""Author: Chih-Yi Lin"""
from classifiers.cnn import *
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from tensorflow import keras
import matplotlib.pyplot as plt


def cnn():
    cnn_model = CNN(embedding_dims=100, filters=128, filter_size=3, hidden_dims=128, batch_size=50, epochs=30)
    # The `removal` parameter has two options: True - remove stopwords and punc ; False - without removal
    cnn_model.preprocess(removal=False)
    cnn_model.create_embedding_matrix('../glove.6B/glove.6B.100d.txt')
    cnn_model.compile()
    cnn_model.fit()
    # Save model
    cnn_model.save_model()
    # Predict
    cnn_model.predict()
    # Evaluation
    print("Detailed Report")
    print(classification_report(cnn_model.y_true, cnn_model.y_pred, target_names=cnn_model.class_name))
    cm = confusion_matrix(cnn_model.y_true, cnn_model.y_pred)
    cmd_obj = ConfusionMatrixDisplay(cm, display_labels=cnn_model.class_name)
    cmd_obj.plot()
    cmd_obj.ax_.set(title='Confusion Matrix',
                xlabel='Predicted Emotions',
                ylabel='Actual Emotions')
    plt.show()

    return cnn_model.y_pred

def multi_channels_cnn():
    cnn_model = CNN(embedding_dims=100, filters=128, filter_size=1, hidden_dims=128, batch_size=50, epochs=30)
    cnn_model.preprocess(removal=False)
    cnn_model.create_embedding_matrix('../glove.6B/glove.6B.100d.txt')
    cnn_model.define_multi_channels([1, 2, 3])
    cnn_model.compile_multi_channels()
    cnn_model.fit_multi_channels()
    # Save model
    cnn_model.save_model()
    # Predict
    cnn_model.predict_multi_channels()
    # Evaluation
    print("Detailed Report")
    print(classification_report(cnn_model.y_true, cnn_model.y_pred, target_names=cnn_model.class_name))
    cm = confusion_matrix(cnn_model.y_true, cnn_model.y_pred)
    cmd_obj = ConfusionMatrixDisplay(cm, display_labels=cnn_model.class_name)
    cmd_obj.plot()
    cmd_obj.ax_.set(title='Confusion Matrix',
                xlabel='Predicted Emotions',
                ylabel='Actual Emotions')
    plt.show()

    return cnn_model.y_pred

def load_model(outpath):
    """load saved model"""
    model = keras.models.load_model(outpath, compile=False)
    return model

def y_pred_file(y_pred):
    """write y_pred into a file for error analysis"""
    with open('../y_pred_1_gram.txt', 'w') as fp:
        for i in y_pred:
            fp.write("%s\n" % i)
        print('File Complete!')

# To run a cnn model
cnn()
# Uncomment the following code to run a multi_channels cnn model
# multi_channels_cnn()
