
import stanza

from diaparser.parsers import Parser
from trankit import Pipeline as tpipe
from models import Sentence, Word

class Connector():
    def __init__(self, model, uas_weight, las_weight):
        self.model = model
        self.uas_weight = float(uas_weight) 
        self.las_weight = float(las_weight)


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
   

class TrankitConnector(Connector):

    def __init_model(self, kwargs):
        if "lang" not in kwargs:
            kwargs["lang"] = "ukrainian"
        try:
            kwargs["model"] = tpipe(lang=kwargs['lang'], gpu=True, cache_dir='./cache')
        except Exception as ex:
            print(ex)
        if "model" not in kwargs:
            kwargs["model"] = tpipe(lang=kwargs['lang'], gpu=True, cache_dir='./cache')

    def __init__(self, **kwargs):
        if "model" not in kwargs:
            self.__init_model(kwargs)
        if "uas_weight" not in kwargs:
            kwargs['uas_weight'] = 1
        if "las_weight" not in kwargs:
            kwargs['las_weight'] = 1
        super().__init__(kwargs['model'], kwargs['uas_weight'], kwargs['las_weight'])

    def predict(self, text):
        processed_sent = self.model(text)
        result = []
        for index_sentence, sentence in enumerate(processed_sent['sentences']):
            counter = 0
            parsed_sentence = Sentence()
            for item in sentence['tokens']:
                counter += 1
                word = Word()
                word.id = item['id']
                word.text = item['text']
                word.lemma = item['lemma']
                word.upos = item['upos']
                word.xpos = item['xpos']
                word.feats = item.get('feats', "_")
                word.head = item['head']
                word.deprel = item['deprel']
                word.misc = '_'
                word.uas_weight = self.uas_weight
                word.las_weight = self.las_weight
                parsed_sentence.add(word)
            result.append(parsed_sentence)
        return result

class DiaConnector(Connector):

    def __init_model(self, kwargs):
        if "corpus_name" not in kwargs:
            kwargs["corpus_name"] = "uk_iu.turkunlp"
        if "lang" not in kwargs:
            kwargs["lang"] = "uk"
        self.language = kwargs["lang"]
        try:
            kwargs["model"] = Parser.load(kwargs["corpus_name"])
        except Exception as ex:
            print(ex)
        if "model" not in kwargs:
            kwargs["model"] = Parser.load(kwargs["corpus_name"])

    def __init__(self, **kwargs):
        if "model" not in kwargs:
            self.__init_model(kwargs)
        if "uas_weight" not in kwargs:
            kwargs['uas_weight'] = 1
        if "las_weight" not in kwargs:
            kwargs['las_weight'] = 1
        super().__init__(kwargs['model'], kwargs['uas_weight'], kwargs['las_weight'])

    def generate_words(self, annotations):
        annotation_list = list(annotations.values())
        result = []
        to_skip = []
        counter = 0
        for item in annotation_list:
            if '# sent_id' in item or '# text' in item:
                continue
            split = item.split('\t')
            if len(split) < 2:
                continue

            id = split[0]
            word = Word()
            if '-' not in id and id not in to_skip:
                counter += 1
                word.id = counter
                word.text = split[1]
                result.append(word)
                continue

            if id in to_skip:
                continue

            ids = id.split('-')
            min = ids[0]
            max = ids[len(ids)-1]
            try: 
                min = int(min)
                max = int(max)
            except ValueError:
                continue

            for number in range(min,max+1):
                to_skip.append(str(number))
            counter += 1
            word.id = counter
            word.text = split[1]
            word.lemma = split[1]
            result.append(word)
        return result

    def predict(self, text):
        dataset = self.model.predict(text, text=self.language)
        result = []
       
        for index_sentence, sentence in enumerate(dataset.sentences):
            words = self.generate_words(sentence.annotations)
            parsed_sentence = Sentence()
            for index, item in enumerate(sentence.words):
                for word_index,word_item in enumerate(words):
                    if word_index > index:
                        continue
                    if words[word_index].text != item:
                        continue
                    words[word_index].upos = '_'
                    words[word_index].xpos = '_'
                    words[word_index].feats = '_'
                    words[word_index].misc = '_'
                    words[word_index].deprel = sentence.rels[index]
                    words[word_index].head = sentence.arcs[index]
                    words[word_index].uas_weight = self.uas_weight
                    words[word_index].las_weight = self.las_weight
            parsed_sentence.words = words
            result.append(parsed_sentence)
        return result