class EmotionClass:

    def __init__(self, label, tokenized_docs):
        self.label = label
        self.docs = tokenized_docs
        # self.weights = {'token': weight}
        self.weights = self.__initial_weights()

    def __initial_weights(self):
        weights = dict()
        weights['BIAS'] = 0
        for doc in self.docs:
            tokens = doc[1]
            for token in tokens:
                weights[token] = 0
        return weights

    def weighted_sum(self, doc):
        """Sum of x*w"""
        sum = 0
        tokens = doc[1]
        for token in tokens:
            if token in self.weights:
                weighted_x = self.weights[token]
                sum += weighted_x
            # When calculate on validation data, there will be tokens not in weights
            else:
                continue
        return sum
