import stanza

from models import Word
from connector import Connector

class StanzaConnector(Connector):

    def __init_model(self, kwargs):
        if "lang" not in kwargs:
            kwargs["lang"] = "uk"
        if "processors" not in kwargs:
            kwargs["processors"] = 'tokenize,mwt,pos,lemma,depparse'

        try:
            kwargs["model"] = stanza.Pipeline(lang=kwargs['lang'], processors=kwargs["processors"])
        except stanza.pipeline.core.LanguageNotDownloadedError:
            stanza.download(kwargs["lang"])
        if "model" not in kwargs:
            kwargs["model"] = stanza.Pipeline(lang=kwargs['lang'], processors=kwargs["processors"])

    def __init__(self, **kwargs) -> None:
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
        for sent in document.sentences:
            for item in sent.words:
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
                result.append(word)
        return result

if __name__ == "__main__":
    connector = StanzaConnector()
    predictions = connector.predict("Зречення культурної ідентичності – це втрата свободи й самовладності.")
    #predictions = classifier.predict_full_text(full_text, "uk")
   