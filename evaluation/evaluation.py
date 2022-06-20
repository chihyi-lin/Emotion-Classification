# TODO: 1. confusion matrix 2. change input data structure
class Evaluation:
    """
    Computing precision, recall and f-score by comparing gold labels and predicted labels in the same list of documents
    """

    def __init__(self, docs):
        # self.eval is a counter for 'tp', 'fp', 'tn', 'fn' for each class= {'joy': {'tp': 333, 'fp': 222, 'tn':2}...}
        self.eval = {}
        self.docs = docs
        self.calculate_tp_fp_fn()

    def calculate_tp_fp_fn(self) -> dict:
        for doc in self.docs:
            # when gold label = predicted label, eval['joy']['tp'] +1
            gold = doc[0]
            predicted = doc[3]
            if gold == predicted:
                self.eval[gold] = self.eval.get(gold, {})
                self.eval[gold]['tp'] = self.eval[gold].get('tp', 0) + 1
            # when gold label != predicted label, eval[predicted label]['fp'] +1, eval[gold label]['fn'] +1
            if gold != predicted:
                self.eval[predicted] = self.eval.get(predicted, {})
                self.eval[predicted]['fp'] = self.eval[predicted].get('fp', 0) + 1
                self.eval[gold] = self.eval.get(gold, {})
                self.eval[gold]['fn'] = self.eval[gold].get('fn', 0) + 1
        return self.eval

    def get_tp(self, label):
        if 'tp' in self.eval[label]:
            return self.eval[label]['tp']
        else:
            return 0

    def get_fp(self, label):
        if 'fp' in self.eval[label]:
            return self.eval[label]['fp']
        else:
            return 0

    def get_fn(self, label):
        if 'fn' in self.eval[label]:
            return self.eval[label]['fn']
        else:
            return 0

    def precision(self, label):
        tp = self.eval[label].get('tp', 0)
        fp = self.eval[label].get('fp', 0)
        precision = tp / (tp + fp)
        return precision

    def recall(self, label):
        tp = self.eval[label].get('tp', 0)
        fn = self.eval[label].get('fn', 0)
        recall = tp / (tp + fn)
        return recall

    def perclass_f_score(self, label):
        precision = self.precision(label)
        recall = self.recall(label)
        perclass_f_score = 2 * precision * recall / (precision + recall)
        return perclass_f_score

    def macro_average(self):
        macro_average = (self.perclass_f_score('joy') +
                         self.perclass_f_score('fear') +
                         self.perclass_f_score('guilt') +
                         self.perclass_f_score('anger') +
                         self.perclass_f_score('shame') +
                         self.perclass_f_score('disgust') +
                         self.perclass_f_score('sadness')) / 7
        return macro_average

    def tps(self):
        tps = (self.get_tp('joy') +
               self.get_tp('fear') +
               self.get_tp('guilt') +
               self.get_tp('anger') +
               self.get_tp('shame') +
               self.get_tp('disgust') +
               self.get_tp('sadness'))
        return tps

    def fps(self):
        fps = (self.get_fp('joy') +
               self.get_fp('fear') +
               self.get_fp('guilt') +
               self.get_fp('anger') +
               self.get_fp('shame') +
               self.get_fp('disgust') +
               self.get_fp('sadness'))
        return fps

    def fns(self):
        fns = (self.get_fn('joy') +
               self.get_fn('fear') +
               self.get_fn('guilt') +
               self.get_fn('anger') +
               self.get_fn('shame') +
               self.get_fn('disgust') +
               self.get_fn('sadness'))
        return fns

    def micro_average(self):
        micro_average = self.tps() / (self.tps() + ((self.fps() + self.fns()) * 0.5))
        return micro_average
