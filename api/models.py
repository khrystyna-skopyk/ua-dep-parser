class Word:
    def __init__(self):
        self.upos = ""
        self.head = ""
        self.deprel = ""
        self.id = ""
        self.lemma = ""
        self.xpos = ""
        self.feats = ""
        self.misc = ""
        self.text = ""
        self.uas_weight = 0
        self.las_weight = 0


class Sentence:
    def __init__(self):
        self.words = []

    def add(self, word):
        self.words.append(word)