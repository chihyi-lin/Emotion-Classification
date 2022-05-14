from corpus import *
from multi_class_perceptron import *
from evaluation import *


class Workflow:

    def __init__(self, iteration):
        self.training_data = self.preprocessing('isear-train.csv')
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
        m = MultiClassPerceptron(self.training_data, self.iteration)
        m.update_weights()
        m.append_prediction()
        return m.docs

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

    def preprocessing_val_data(self):
        val_data = self.preprocessing('isear-val.csv')
        return val_data

    def make_prediction_on_val_data(self):
        val_data = self.preprocessing_val_data()
        for doc in val_data:
            find_max = dict()
            for perceptron in MultiClassPerceptron.all_perceptrons:
                find_max[perceptron.label] = perceptron.weighted_sum(doc)
            predicted = max(find_max, key=find_max.get)
            doc.append(predicted)
        return val_data


w = Workflow(20)
docs = w.training()
w.evaluation(docs)  # Print out f-score on training data
w.preprocessing_val_data()
val_docs = w.make_prediction_on_val_data()
w.evaluation(val_docs)  # Print out f-score on validation data
