import stanza

from models import Word, Sentence
from connector import Connector


class StanzaConnector(Connector):

    def __init_model(self, kwargs):
        if "lang" not in kwargs:
            kwargs["lang"] = "uk"
        if "processors" not in kwargs:
            kwargs["processors"] = 'tokenize,mwt,pos,lemma,depparse'

        try:
            kwargs["model"] = stanza.Pipeline(lang=kwargs['lang'], processors=kwargs["processors"])
        except Exception as ex:
            stanza.download(kwargs["lang"])
        if "model" not in kwargs:
            kwargs["model"] = stanza.Pipeline(lang=kwargs['lang'], processors=kwargs["processors"])

    def __init__(self, **kwargs):
        if "model" not in kwargs:
            self.__init_model(kwargs)
        if "uas_weight" not in kwargs:
            kwargs['uas_weight'] = 1
        if "las_weight" not in kwargs:
            kwargs['las_weight'] = 1
        super().__init__(kwargs['model'], kwargs['uas_weight'], kwargs['las_weight'])

    def predict(self, text):
        document = self.model(text)
        result = []
        for index_sentence, sentence in enumerate(document.sentences):
            parsed_sentence = Sentence()
            for item in sentence.words:
                word = Word()
                word.id = item.id
                word.text = item.text
                word.lemma = item.lemma
                word.upos = item.upos
                word.xpos = item.xpos
                word.feats = item.feats
                word.head = item.head
                word.deprel = item.deprel
                word.misc = item.misc
                word.uas_weight = self.uas_weight
                word.las_weight = self.las_weight
                parsed_sentence.add(word)
            result.append(parsed_sentence)
        return result
   