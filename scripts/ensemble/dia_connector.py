from diaparser.parsers import Parser

from models import Word
from connector import Connector

class DiaConnector(Connector):

    def __init_model(self, kwargs):
        if "corpus_name" not in kwargs:
            kwargs["corpus_name"] = "uk_iu.TurkuNLP"
        if "lang" not in kwargs:
            kwargs["lang"] = "uk"
        self.language = kwargs["lang"]
        try:
            kwargs["model"] = Parser.load(kwargs["corpus_name"])
        except Exception as ex:
            print(ex)
        if "model" not in kwargs:
            kwargs["model"] = Parser.load(kwargs["corpus_name"])

    def __init__(self, **kwargs) -> None:
        if "model" not in kwargs:
            self.__init_model(kwargs)
        if "uas_weight" not in kwargs:
            kwargs['uas_weight'] = 1
        if "las_weight" not in kwargs:
            kwargs['las_weight'] = 1
        super().__init__(kwargs['model'], kwargs['uas_weight'], kwargs['las_weight'])

    def predict(self, text):
        dataset = self.model.predict(text, text=self.language)
        result = []
        id_count = 0
        sentence = dataset.sentences[0]
        for index, item in enumerate(sentence.words):
            id_count += 1

            word = Word()
            word.id = id_count
            word.text = item
            word.upos = None
            word.deprel = sentence.rels[index]
            word.head = sentence.arcs[index]
            word.uas_weight = self.uas_weight
            word.las_weight = self.las_weight
            result.append(word)

        return result


    