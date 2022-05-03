from corpus import *


class Perceptron:

    doc = Document()
    doc = doc.tokenized_text

    def __init__(self):
        self.weights = self.initial_weights()

    def initial_weights(self):
        weights = dict()
        for line in self.doc:
            tokens = line[1]
            for token in tokens:
                weights[token] = 0.1
        return weights

    def prediction(self):
        for doc in self.doc:
            sum = 0
            tokens = doc[1]
            for token in tokens:
                weighted_x = self.weights[token]
                sum += weighted_x
            doc.append(sum)
        return self.doc

    def update_weights(self):
        for i in range(10):
            for doc in self.doc:
                gold = doc[0]
                predict = doc[-1]
                tokens = doc[1]
                if gold == 'joy' and predict > 0:
                    pass
                if gold != 'joy' and predict < 0:
                    pass
                if gold == 'joy' and predict < 0:
                    for token in tokens:
                        self.weights[token] = self.weights[token]+0.1
                        #?correct prediction?
                if gold != 'joy' and predict > 0:
                    for token in tokens:
                        self.weights[token] = self.weights[token]-0.1
        # print keys which have value > 0 -> keys which belongs to 'joy'
        # for k,v in self.weights.items():
        #     if v > 0:
        #         print(k)




p = Perceptron()
p.prediction()
p.update_weights()
score_for_happy = p.weights['happy']
score_for_wonderful = p.weights['wonderful']
score_for_school = p.weights['school']
print(score_for_happy, score_for_wonderful, score_for_school)
