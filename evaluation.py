from corpus import *


class MakePrediction:

    def __init__(self, document):
        self.document = document

    def assign_predicted_label(self) -> list:
        """
        *Later will be replaced by the classifier.*
        Dividing the original dataset into 7 batches, assigning one emotion to each batch: 765 / 765 / 765 / 765 / 765 / 765 / 765
        :return: list of texts with predicted labels
        """
        predict = [x[:] for x in self.document]
        for j in predict[:765]:
            j[0] = 'joy'
        for j in predict[765:765*2]:
            j[0] = 'fear'
        for j in predict[765*2:765*3]:
            j[0] = 'guilt'
        for j in predict[765*3:765*4]:
            j[0] = 'anger'
        for j in predict[765*4:765*5]:
            j[0] = 'shame'
        for j in predict[765*5:765*6]:
            j[0] = 'disgust'
        for j in predict[765*6:]:
            j[0] = 'sadness'
        return predict


class Evaluation:

    def __init__(self, document, predict):
        # self.eval is a counter for 'tp', 'fp', 'tn', 'fn' for each class= {'joy': {'tp': 333, 'fp': 222, 'tn':2}...}
        self.eval = {}
        self.document = document
        self.predict = predict

    def __calculate_tp_fp_fn(self) -> dict:
        for g, p in zip(self.document, self.predict):
            # when gold label = predicted label, eval['joy']['tp'] +1
            gold_label = g[0]
            predicted_label = p[0]
            if gold_label == predicted_label:
                self.eval[gold_label] = self.eval.get(gold_label, {})
                self.eval[gold_label]['tp'] = self.eval[gold_label].get('tp', 0) + 1
            # when gold label != predicted label, eval[predicted label]['fp'] +1, eval[gold label]['fn'] +1
            if gold_label != predicted_label:
                self.eval[predicted_label] = self.eval.get(predicted_label, {})
                self.eval[predicted_label]['fp'] = self.eval[predicted_label].get('fp', 0) + 1
                self.eval[gold_label] = self.eval.get(gold_label, {})
                self.eval[gold_label]['fn'] = self.eval[gold_label].get('fn', 0) + 1
        return self.eval

    def precision(self, label):
        self.__calculate_tp_fp_fn()
        tp = self.eval[label]['tp']
        fp = self.eval[label]['fp']
        precision = tp / (tp + fp)
        return precision

    def recall(self, label):
        self.__calculate_tp_fp_fn()
        tp = self.eval[label]['tp']
        fn = self.eval[label]['fn']
        recall = tp / (tp + fn)
        return recall

    def f_score(self, label):
        precision = self.precision(label)
        recall = self.recall(label)
        f_score = 2 * precision * recall / (precision + recall)
        return f_score


def run_evaluation():
    c = Corpus("isear-train.csv")
    c.read_file()
    c.remove_invalid_texts_and_assign_text_id()
    c = MakePrediction(c.document)
    predict = c.assign_predicted_label()
    eval_1 = Evaluation(c.document, predict)
    f_score_joy = eval_1.f_score('joy')
    f_score_fear = eval_1.f_score('fear')
    f_score_guilt = eval_1.f_score('guilt')
    f_score_anger = eval_1.f_score('anger')
    f_score_shame = eval_1.f_score('shame')
    f_score_disgust = eval_1.f_score('disgust')
    f_score_sadness = eval_1.f_score('sadness')
    print("f score for joy: {} \n"
          "f score for fear: {} \n"
          "f score for guilt: {} \n"
          "f score for anger: {} \n"
          "f score for shame: {} \n"
          "f score for disgust: {} \n"
          "f score for sadness: {} \n"
          .format(f_score_joy, f_score_fear, f_score_guilt, f_score_anger, f_score_shame, f_score_disgust, f_score_sadness))


run_evaluation()