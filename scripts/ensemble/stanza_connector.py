import stanza

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
        results = {}
        index = 0
        for sent in document.sentences:
            for word in sent.words:
                parsing_result = {}
                parsing_result["id"] = word.id
                parsing_result["upos"] = str.lower(word.upos)
                parsing_result["head"] = word.head
                results[index] = parsing_result
                index += 1
        return results



    