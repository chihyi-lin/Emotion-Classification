class Evaluation:
    """
    Computing precision, recall and f-score by comparing gold labels and predicted labels in the same list of documents
    """

    def __init__(self, docs):
        # self.eval is a counter for 'tp', 'fp', 'tn', 'fn' for each class= {'joy': {'tp': 333, 'fp': 222, 'tn':2}...}
        self.eval = {}
        self.docs = docs

    def __calculate_tp_fp_fn(self) -> dict:
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
