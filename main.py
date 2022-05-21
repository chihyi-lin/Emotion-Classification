
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
              "f score in general(macro average): {} \n"
              .format(f_score_joy, f_score_fear, f_score_guilt, f_score_anger, f_score_shame, f_score_disgust, f_score_sadness,(f_score_joy + f_score_fear + f_score_guilt + f_score_anger + f_score_shame + f_score_disgust + f_score_sadness)/7))

    def accuracy(self, docs_with_predicted_label):
        acc = Evaluation(docs_with_predicted_label)
        tp_joy = acc.tp('joy')
        tp_fear = acc.tp('fear')
        tp_guilt = acc.tp('guilt')
        tp_anger = acc.tp('anger')
        tp_shame = acc.tp('shame')
        tp_disgust = acc.tp('disgust')
        tp_sadness = acc.tp('sadness')
        fp_joy = acc.fp('joy')
        fp_fear = acc.fp('fear')
        fp_guilt = acc.fp('guilt')
        fp_anger = acc.fp('anger')
        fp_shame = acc.fp('shame')
        fp_disgust = acc.fp('disgust')
        fp_sadness = acc.fp('sadness')
        fn_joy = acc.fn('joy')
        fn_fear = acc.fn('fear')
        fn_guilt = acc.fn('guilt')
        fn_anger = acc.fn('anger')
        fn_shame = acc.fn('shame')
        fn_disgust = acc.fn('disgust')
        fn_sadness = acc.fn('sadness')
        print("accuracy(micro average): {} \n"
            .format((tp_joy + tp_fear + tp_guilt + tp_anger + tp_shame + tp_disgust + tp_sadness)/((tp_joy + tp_fear + tp_guilt + tp_anger + tp_shame + tp_disgust + tp_sadness)+ ((fp_joy + fp_fear + fp_guilt + fp_anger + fp_shame + fp_disgust + fp_sadness)+(fn_joy + fn_fear + fn_guilt + fn_anger + fn_shame + fn_disgust + fn_sadness)) * 0.5)))

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
w.accuracy(docs)
w.preprocessing_val_data()
val_docs = w.make_prediction_on_val_data()
w.evaluation(val_docs)  # Print out f-score on validation data
w.accuracy(val_docs)
