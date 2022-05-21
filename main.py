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
              "f score in general(macro average): {} \n"
              .format(f_score_joy, f_score_fear, f_score_guilt, f_score_anger, f_score_shame, f_score_disgust, f_score_sadness, (f_score_joy + f_score_fear + f_score_guilt + f_score_anger + f_score_shame + f_score_disgust + f_score_sadness)/7))

    def accuracy(self, docs_with_predicted_label):
        acc = Evaluation(docs_with_predicted_label)
        tp_joy = acc.get_tp('joy')
        tp_fear = acc.get_tp('fear')
        tp_guilt = acc.get_tp('guilt')
        tp_anger = acc.get_tp('anger')
        tp_shame = acc.get_tp('shame')
        tp_disgust = acc.get_tp('disgust')
        tp_sadness = acc.get_tp('sadness')
        fp_joy = acc.get_fp('joy')
        fp_fear = acc.get_fp('fear')
        fp_guilt = acc.get_fp('guilt')
        fp_anger = acc.get_fp('anger')
        fp_shame = acc.get_fp('shame')
        fp_disgust = acc.get_fp('disgust')
        fp_sadness = acc.get_fp('sadness')
        fn_joy = acc.get_fn('joy')
        fn_fear = acc.get_fn('fear')
        fn_guilt = acc.get_fn('guilt')
        fn_anger = acc.get_fn('anger')
        fn_shame = acc.get_fn('shame')
        fn_disgust = acc.get_fn('disgust')
        fn_sadness = acc.get_fn('sadness')
        print("accuracy(micro average): {} \n"
              .format((tp_joy + tp_fear + tp_guilt + tp_anger + tp_shame + tp_disgust + tp_sadness) / (
                    (tp_joy + tp_fear + tp_guilt + tp_anger + tp_shame + tp_disgust + tp_sadness) + (
                        (fp_joy + fp_fear + fp_guilt + fp_anger + fp_shame + fp_disgust + fp_sadness) + (
                            fn_joy + fn_fear + fn_guilt + fn_anger + fn_shame + fn_disgust + fn_sadness)) * 0.5)))

    def make_prediction_on_val_data(self, multi_class_perceptron:MultiClassPerceptron):
        val_data = multi_class_perceptron.validation_docs
        for doc in val_data:
            tokens = doc[1]
            # append BIAS to the token list
            tokens.append("BIAS")
            predicted = multi_class_perceptron.find_max_prediction(doc)
            doc.append(predicted)
        return val_data


w = Workflow(10)
trained_perceptron = w.training()
print("f scores for training data:")
w.evaluation(trained_perceptron.docs)
w.accuracy(trained_perceptron.docs)  # Print accuracy/micro average
val_docs = w.make_prediction_on_val_data(trained_perceptron)
print("f scores for validation data:")
w.evaluation(val_docs)
w.accuracy(val_docs)