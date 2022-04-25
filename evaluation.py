file_path = "isear-train.csv"

gold = []     # gold = [['1st label', '1st text', 'text1_id'], ['2nd label', '2nd text', 'text2_id']]
with open(file_path, 'r') as f:
    for line in f:
        line = line.replace("\n", "").split(",")
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

# ? How to build counts for 'tp', 'fp', 'tn', 'fn'= {'tp': {'joy': 333, 'fear':222...}, 'fp': {'joy': 2, 'fear': 80}...}?
counts = {}
for line_g in gold:
    for line_p in predict:
        if line_g[0] == line_p[0]:
            l_dict = counts.get('tp', {})
            l_dict[line_p[0]] = 1
        # else:
        #     counts['tp'][line_p[0]] = 1
        #
        # if line_g[0] != line_p[0]:
        #     if 'fp' in counts and line_p[0] in counts['fp']:
        #         counts['fp'][line_p[0]] += 1
        #     elif 'fp' in counts:
        #         counts['fp'][line_p[0]] = 0



