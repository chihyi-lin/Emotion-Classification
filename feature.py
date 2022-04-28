from evaluation import *
import re

# Texts cleaning: lowercase, remove punctuation
for line in gold:
    line[1] = line[1].lower()
    line[1] = re.findall(r'[-\'\w]+', line[1])


# Find most frequent words in each emotion: count token frequency
def find_frequent_words(emotion: str):
    word_counts = {}
    for g in gold:
        if g[0] not in word_counts:
            word_counts[g[0]] = word_counts.get(g[0], {})
        for token in g[1]:
            if token not in word_counts:
                word_counts[g[0]][token] = word_counts[g[0]].get(token, 0)+1
    # Find words which have frequencies between 5-150 in each emotion
    candidate_for_feature = {}
    for k, v in word_counts[emotion].items():
        if 150 > v > 5:
            candidate_for_feature[k] = v
    return candidate_for_feature


# Manually pick some adjectives, verbs and nouns for 'joy' from 'candidate_for_feature'
joy = {'happy', 'happiest', 'good', 'great', 'wonderful', 'nice', 'best', 'successful', 'pleasant', 'loved',
       'admitted', 'accepted', 'selected', 'love', 'visit',
       'university', 'school', 'boyfriend', 'wedding', 'friends', 'friend', 'holiday', 'party', 'christmas', 'birthday', 'relationship', 'experience', 'acceptance'}

class FeatureExtraction:
    """
    - Compare texts with the joy set (or other emotion sets).
    - If texts contain tokens in the emotion set, we assign 1 to 'in_dict_or_not', else assign 0.
    - The numbers in feature vector mean the position which hold value '1' in 'in_dict_or_not', and the position which hold '0' we don't save them.
    E.g., 0 could mean 'contain words in the joy set', 1 could mean 'contain words in the fear set', etc.
    """
    def extract_features(self=gold):
        for line in gold[:20]:
            feature_vector = list()
            in_dict_or_not = set()
            tokens = line[1]
            for token in tokens:
                if token in joy:
                    in_dict_or_not.add(1)
                else:
                    in_dict_or_not.add(0)
    # Feature vector only represents the positions which hold 1
            if 1 in in_dict_or_not:
                feature_vector.append(0)
            line.append(feature_vector)

        return gold[:20]


# result = FeatureExtraction.extract_features()
# print(result)
