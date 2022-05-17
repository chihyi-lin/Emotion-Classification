from corpus import *
from multi_class_perceptron import *
from evaluation import *


class Workflow:

    def __init__(self, iteration):
        self.training_data = self.preprocessing('isear-train.csv')
        self.validation_data = self.preprocessing('isear-val.csv')
        self.iteration = iteration

    # Create documents as a nested list, tokenizing texts
    def preprocessing(self, file: str):
        c = Corpus(file)
        docs = c.documents
        d = Document(docs)
        tokenized_data = d.tokenize()
        return tokenized_data

    # Instantiating MultiClassPerceptron, training on weights for assigned iterations and appending the results
    def training(self):
        m = MultiClassPerceptron(self.training_data, self.validation_data, self.iteration)
        m.update_weights()
        m.append_prediction()
        return m

    def evaluation(self, docs_with_predicted_label):
        eval = Evaluation(docs_with_predicted_label)
        f_score_joy = eval.f_score('joy')
        f_score_fear = eval.f_score('fear')
        f_score_guilt = eval.f_score('guilt')
        f_score_anger = eval.f_score('anger')
        f_score_shame = eval.f_score('shame')
        f_score_disgust = eval.f_score('disgust')
        f_score_sadness = eval.f_score('sadness')
        print("f score for joy: {} \n"
              "f score for fear: {} \n"
              "f score for guilt: {} \n"
              "f score for anger: {} \n"
              "f score for shame: {} \n"
              "f score for disgust: {} \n"
              "f score for sadness: {} \n"
              .format(f_score_joy, f_score_fear, f_score_guilt, f_score_anger, f_score_shame, f_score_disgust, f_score_sadness))

    def make_prediction_on_val_data(self, multi_class_perceptron:MultiClassPerceptron):
        val_data = multi_class_perceptron.validation_docs
        for doc in val_data:
            tokens = doc[1]
            # append BIAS to the token list
            tokens.append("BIAS")
            predicted = multi_class_perceptron.find_max_prediction(doc)
            doc.append(predicted)
        return val_data


w = Workflow(30)
trained_perceptron = w.training()
print("f scores for training data:")
w.evaluation(trained_perceptron.docs)
val_docs = w.make_prediction_on_val_data(trained_perceptron)
print("f scores for validation data:")
w.evaluation(val_docs)
