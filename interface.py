class Workflow:
    def main(self):
        documents = self.preprocessing("isear-train.csv")
        classifier = self.training(documents)
        self.test(classifier)

    def preprocessing(self, file_name: str):
        corpus = Corpus(file_name)
        documents = corpus.read_file()
        for document in documents:
            document.tokens = Tokenizer().tokenize(document.text)
            document.features = FeatureExtraction().extract_features(document.tokens)
        return documents

    def training(self, documents: list):
        return Classifier()

    def test(self, classifier):
        documents = self.preprocessing("isear-val.csv")
        for document in documents:
            classifier.classify(document)

    def evaluation(self):
        pass

class Corpus:
    """
    Read file
    :return: evaluation
    """
    def __init__(self, file_name: str):
        self.file_name = file_name
    def read_file(self) -> list:
        documents = list()
        documents.append(Document("gold_label", "txt"))
        return documents

    def evaluation(self):
        pass

class Document:
    """
    Attributes: Text ID, text, list of tokens, gold label, predicted label, features
    :return:
    """
    def get_tokens(self) -> list:
        return self.tokens

    def __init__(self, gold_label: str, text: str):
        self.gold_label = gold_label
        self.text = text
        self.tokens = list()
        self.features = list()

class FeatureExtraction:
    def extract_features(self, tokens: list):
        features = list()
        features.append(Feature())
        return features

class Feature:
    pass
class Token:
    """

    """
    pass

class Tokenizer:
    """
    Clean and Tokenize a text, has a normalized string.
    :return: list of tokens
    """
    def tokenize(self, document) -> list:
        tokens = list()
        tokens.append(Token())
        return tokens
    pass

class Classifier:
    def __init__(self):
        self.emotions = list()
        self.emotions.append("joy")
        self.emotions.append("guilt")
        self.emotions.append("sadness")
        self.emotions.append("anger")
        self.emotions.append("shame")
        self.emotions.append("disgust")
        self.emotions.append("fear")

    def classify(self, document) -> str:
        features = document.features
        return "joy"

class Evaluator:
    """

    """
    pass