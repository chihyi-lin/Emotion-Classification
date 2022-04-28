file_path = "isear-train.csv"

gold = []     # gold = [['1st label', '1st text', 'text1_id'], ['2nd label', '2nd text', 'text2_id']]
with open(file_path, 'r') as f:
    for line in f:
        line = line.replace("\n", "").split(",", 1)
        gold.append(line)

# Drop texts without label: 10 has been removed. len(data_list)=5355
# Assign text_id to each text, 5345 text_id in total (?Why not = 5355?)
label_dict = {'joy', 'fear', 'guilt', 'anger', 'shame', 'disgust', 'sadness'}
text_id = 1
for i in gold:
    if i[0] not in label_dict:
        gold.remove(i)
    else:
        i.append(text_id)
        text_id += 1

# Create list of predicted label with each class evenly distributed: 765 / 765 / 765 / 765 / 765 / 765 / 765
predict = [x[:] for x in gold]
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


class Evaluation:
    def __init__(self):
        self.eval = {}

# Build counts for 'tp', 'fp', 'tn', 'fn' for each class= {'joy': {'tp': 333, 'fp': 222, 'tn':2}...}
    def __calculate_tp_fp_fn(self):
        for g, p in zip(gold, predict):
            # when gold label = predicted label, eval['joy']['tp'] +1
            if g[0] == p[0]:
                self.eval[g[0]] = self.eval.get(g[0], {})
                self.eval[g[0]]['tp'] = self.eval[g[0]].get('tp', 0) + 1
            # when gold label != predicted label, eval[predicted label]['fp'] +1, eval[gold label]['fn'] +1
            if g[0] != p[0]:
                self.eval[p[0]] = self.eval.get(p[0], {})
                self.eval[p[0]]['fp'] = self.eval[p[0]].get('fp', 0) + 1
                self.eval[g[0]] = self.eval.get(g[0], {})
                self.eval[g[0]]['fn'] = self.eval[g[0]].get('fn', 0) + 1

    def f_score(self, label):
        self.__calculate_tp_fp_fn()
        precision = self.eval[label]['tp'] / (self.eval[label]['tp'] + self.eval[label]['fp'])
        recall = self.eval[label]['tp'] / (self.eval[label]['tp'] + self.eval[label]['fn'])
        f_score = 2 * precision * recall / (precision + recall)
        return f_score

# e = Evaluation()
# print(e.f_score('joy'))
# print(e.f_score('fear'))
# print(e.f_score('shame'))
# print(e.f_score('disgust'))
# print(e.f_score('guilt'))
# print(e.f_score('anger'))
# print(e.f_score('sadness'))
