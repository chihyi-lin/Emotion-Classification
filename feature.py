from corpus import *


class FeatureExtraction:

    # Manually pick some adjectives, verbs and nouns for 'joy' from 'candidate_for_feature'
    joy = {'happy', 'happiest', 'good', 'great', 'wonderful', 'nice', 'best', 'successful', 'pleasant', 'loved',
           'admitted', 'accepted', 'selected', 'love', 'visit',
           'university', 'school', 'boyfriend', 'wedding', 'friends', 'friend', 'holiday', 'party', 'christmas',
           'birthday', 'relationship', 'experience', 'acceptance'}

    def __init__(self, document):
        self.document = document

    # Find most frequent words in each emotion: count token frequency
    def find_frequent_words(self, emotion: str):
        word_counts = {}
        for g in self.document:
            label = g[0]
            if label not in word_counts:
                word_counts[label] = word_counts.get(label, {})
            tokens = g[1]
            for token in tokens:
                if token not in word_counts:
                    word_counts[label][token] = word_counts[label].get(token, 0) + 1
        # Find words which have frequencies between 5-150 in each emotion to filter out stop words
        candidate_for_feature = {}
        for k, v in word_counts[emotion].items():
            if 150 > v > 5:
                candidate_for_feature[k] = v
        return candidate_for_feature

    def get_feature_vector(self) -> list:
        """
        - Compare each text with the joy set (or other emotion sets).
        - If the text contains tokens in the emotion set, we assign 1 to 'in_dict_or_not', else assign 0.
        - The numbers in feature vector mean the position which hold value '1' in 'in_dict_or_not', and the position which hold '0' we don't save them.
        E.g., 0 could mean 'contain words in the joy set', 1 could mean 'contain words in the fear set', etc.
        """
        for line in self.document:
            feature_vector = list()
            in_dict_or_not = set()
            tokens = line[1]
            for token in tokens:
                if token in self.joy:
                    in_dict_or_not.add(1)
                else:
                    in_dict_or_not.add(0)
    # Feature vector only represents the positions which hold 1
            if 1 in in_dict_or_not:
                feature_vector.append(0)
            line.append(feature_vector)

        return self.document


def tokenize_and_extract_features():
    c = Corpus("isear-train.csv")
    c.read_file()
    c.remove_invalid_texts_and_assign_text_id()
    c_tokenize = c.tokenize()
    return c_tokenize


def get_vector():
    c_tokenize = tokenize_and_extract_features()
    c_feature_vector = FeatureExtraction(c_tokenize[:20]).get_feature_vector()
    return c_feature_vector


print(get_vector())

