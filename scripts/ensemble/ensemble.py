import operator
from typing import Sequence
import stanza
import time

from models import Word, Sentence
from stanza_connector import StanzaConnector
from dia_connector import DiaConnector
from collections import OrderedDict
from conllu.models import TokenList, Token
from conllu import parse_tree

class DependencyParsingClassifier:
    def __init__(self, connectors):
        self.connectors = connectors
        self.sentences = []

    def append(self, connector):
        if connector != None:
            self.connectors.append(connector)

    def revert_predictions(self, predictions):
        if len(predictions) == 0:
            return []
        batches = [[] for number in range(len(predictions[0]))]

        for predictions_list in predictions:
            for index_prediction, prediction in enumerate(predictions_list):
                if len(batches[index_prediction]) == 0:
                    batches[index_prediction] = [[] for number in range(len(prediction.words))]
                for index, word in enumerate(prediction.words):
                    batches[index_prediction][index].append(word)
        return batches

    def merge_predictions(self, predictions):
        parsed_sentence = Sentence()
        for sentence in predictions:
            for word_list in sentence:
                deprels = {} 
                heads = {}
                ids = {}
                texts= {}
                uposes = {}
                lemmas = {}
                xposes = {}
                feats_list = {}
                miscs = {}
                for word in word_list:
                    self.create_values_dict(word.deprel, deprels, word.las_weight)
                    self.create_values_dict(word.head, heads, word.uas_weight)
                    self.create_values_dict(word.id, ids, word.uas_weight)
                    self.create_values_dict(word.text, texts, word.uas_weight)
                    self.create_values_dict(word.upos, uposes, word.uas_weight)
                    self.create_values_dict(word.lemma, lemmas, word.uas_weight)
                    self.create_values_dict(word.xpos, xposes, word.uas_weight)
                    self.create_values_dict(word.feats, feats_list, word.uas_weight)
                    self.create_values_dict(word.misc, miscs, word.uas_weight)

                word = self.merge_word_values(deprels, heads, ids, texts, uposes, lemmas,xposes, feats_list, miscs)
                parsed_sentence.add(word)
        return parsed_sentence

    def merge_word_values(self, deprels, heads, ids, texts, uposes, lemmas, xposes, feats_list, miscs):
        id = self.merge_dict_values(ids)
        head = self.merge_dict_values(heads)
        deprel = self.merge_dict_values(deprels)
        text = self.merge_dict_values(texts)
        upos = self.merge_dict_values(uposes)
        lemma = self.merge_dict_values(lemmas)
        xpos = self.merge_dict_values(xposes)
        feats = self.merge_dict_values(feats_list)
        misc = self.merge_dict_values(miscs)

        word = Word()
        word.id = id
        word.text = text
        word.lemma = lemma
        word.upos = upos
        word.xpos = xpos
        word.feats = feats
        word.head = head
        word.deprel = deprel
        word.misc = misc
        return word


    def merge_dict_values(self, input_dict):
        selection_dict = {}
        for item in input_dict:
            if item not in selection_dict:
                selection_dict[item] = 0
            selection_dict[item] += input_dict[item]["weight"] * input_dict[item]["count"]
        return max(selection_dict.items(), key=operator.itemgetter(1))[0]

    def create_values_dict(self, value, input_dict, weigth):

        if value == None:
            value = ""
            
        if value not in input_dict:
            input_dict[value] = {}
            input_dict[value]["count"] = 0
            input_dict[value]["weight"] = 0

        input_dict[value]["count"] += 1
        input_dict[value]["weight"] += weigth

    def predict_full_text(self,text, delay = 0):
        sentences = text.split('\n')
        parsed_sentences = []
        for sentence in sentences:
            time.sleep(delay)
            print(sentence)
            parsed_sentence = self.predict(sentence)
            parsed_sentences.append(parsed_sentence)
        self.sentences = parsed_sentences
        return parsed_sentences

    def predict(self, sentence):
        predictions = []
        for connector in self.connectors:
            prediction = connector.predict(sentence)
            predictions.append(prediction)
        predictions = self.revert_predictions(predictions)
        parsed_sentence = self.merge_predictions(predictions)
        return parsed_sentence

    def write_to_conllu(self, path):
        sentences_to_write = [] 
        for sentence_item in self.sentences:
            token_list = TokenList()
            for word in sentence_item.words:
                compiled_tokens = OrderedDict({'id': word.id, 'form': word.text, 'lemma': word.lemma, 'upos': word.upos,
                 'xpos': word.xpos, 'feats':word.feats, 'head': word.head, 'deprel': word.deprel, 'headdeprel':f'{word.head}:{word.deprel}','misc':word.misc})
                token_list.append(compiled_tokens)
            sentences_to_write.append(token_list)

        with open(path, 'w') as file:
            file.writelines([sentence.serialize() for sentence in sentences_to_write])



if __name__ == "__main__":
    connector1 = StanzaConnector()
    #connector2 = DiaConnector()

    with open('uk_iu-ud-test.txt') as f:
        full_text = f.read()

    classifier = DependencyParsingClassifier([connector1])
    predictions = classifier.predict_full_text(full_text)
    print(predictions)
    classifier.write_to_conllu("ensemble.conllu")