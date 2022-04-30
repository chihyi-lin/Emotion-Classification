import re


class Corpus:

    # self.document = [['1st label', '1st text', 'text1_id'], ['2nd label', '2nd text', 'text2_id']]
    def __init__(self, file_name):
        self.file_name = file_name
        self.document = list()

    def read_file(self) -> list:
        with open(self.file_name, 'r') as f:
            for line in f:
                line = line.replace("\n", "").split(",", 1)
                self.document.append(line)
        return self.document

    def remove_invalid_texts_and_assign_text_id(self):
        # Drop texts without labels: 10 have been removed. len(data_list)=5355
        # Assign text_id to each text, 5345 text_id in total (?Why not = 5355?)
        labels = {'joy', 'fear', 'guilt', 'anger', 'shame', 'disgust', 'sadness'}
        text_id = 1
        for line in self.document:
            if line[0] not in labels:
                self.document.remove(line)
            else:
                line.append(text_id)
                text_id += 1
        return self.document

    def tokenize(self):
        """Convert texts into lowercase, remove punctuations, tokenize texts"""
        for line in self.document:
            line[1] = line[1].lower()
            line[1] = re.findall(r'[-\'\w]+', line[1])
        return self.document
