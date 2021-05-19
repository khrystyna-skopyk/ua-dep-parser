import operator
from typing import Sequence
import stanza
from stanza.models.common.pretrain import Pretrain
import time


from models import Word, Sentence
from helpers import Graph
from stanza_connector import StanzaConnector
from dia_connector import DiaConnector
from trankit_connector import TrankitConnector
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
        parsed_sentences = []
        for sentence in predictions:
            parsed_sentence = Sentence()
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
            parsed_sentences.append(parsed_sentence)
        return parsed_sentences

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
            parsed_sentences += self.predict(sentence)
        self.sentences = parsed_sentences
        return parsed_sentences

    def predict(self, sentence):
        predictions = []
        for connector in self.connectors:
            prediction = connector.predict(sentence)
            predictions.append(prediction)
        predictions = self.revert_predictions(predictions)
        parsed_sentences = self.merge_predictions(predictions)
        for sentence in parsed_sentences:
            self.check_for_circle(sentence)
        return parsed_sentences

    def check_for_circle(self, sentence):
        relations = {}
        for word in sentence.words:
            if word.head not in relations:
                relations[word.head] = []
            relations[word.head].append(word.id)

        graph = Graph(len(sentence.words)+1)
        for relation in relations:
            connections = relations[relation]
            for connection in connections:
                graph.add_edge(relation, connection)
        print(graph.check_is_cyclic())


    def write_to_conllu(self, path):
        sentences_to_write = [] 
        for sentence_item in self.sentences:
            token_list = TokenList()
            for word in sentence_item.words:
                compiled_tokens = OrderedDict()
                compiled_tokens['id'] = word.id
                compiled_tokens['form'] = word.text
                compiled_tokens['lemma'] = word.lemma
                compiled_tokens['upos'] = word.upos
                compiled_tokens['xpos'] = word.xpos
                compiled_tokens['feats'] = word.feats
                compiled_tokens['head'] = word.head
                compiled_tokens['deprel'] = word.deprel
                compiled_tokens['headdeprel'] = str(word.head) +":" + str(word.deprel)
                compiled_tokens['misc'] = word.misc
                token_list.append(compiled_tokens)
            sentences_to_write.append(token_list)

        with open(path, 'w') as file:
            file.writelines([sentence.serialize() for sentence in sentences_to_write])



if __name__ == "__main__":
    
    with open('uk_iu-ud-test.txt') as f:
        full_text = f.read()

    full_text = "Супроти тамошнього населення - культурна, ввічлива, бадьора, весела, - як пристало на синів культурного і лицарського 45-ти мільйонового українського народу та воїнів вкритої славою революційної УПА."
    
    pt_original = Pretrain("ewt_original.pt", "./models/original/ukoriginalvectors.xz")
    pt_fast_text = Pretrain("ewt_fast_text.pt", "./models/fast-text/uk.vectors.xz")
    pt_glove = Pretrain("ewt_glove.pt", "./models/glove/glove.xz")
 
    pt_original.load()
    pt_fast_text.load()
    pt_glove.load()

    config_original = {
        'processors': 'pos, lemma, tokenize, depparse',
        'lang': 'uk',
        'depparse_model_path': './models/original/depparse/uk_iu_parser.pt',
        'pos_pretrain_path': 'ewt_original.pt',
        'depparse_pretrain_path': 'ewt_original.pt',
        'tokenize_model_path': './models/original/tokenize/uk_iu_tokenizer.pt',
        'pos_model_path': './models/original/pos/uk_iu_tagger.pt',
        'lemma_model_path': './models/original/lemma/uk_iu_lemmatizer.pt',
        'mwt_model_path': './models/original/mwt/uk_iu_mwt_expander.pt'
    }

    config_fast_text = {
        'processors': 'pos, lemma, tokenize, depparse',
        'lang': 'uk',
        'depparse_model_path': './models/fast-text/depparse/uk_iu_parser.pt',
        'pos_pretrain_path': 'ewt_fast_text.pt',
        'depparse_pretrain_path': 'ewt_fast_text.pt',
        'tokenize_model_path': './models/fast-text/tokenize/uk_iu_tokenizer.pt',
        'pos_model_path': './models/fast-text/pos/uk_iu_tagger.pt',
        'lemma_model_path': './models/fast-text/lemma/uk_iu_lemmatizer.pt',
        'mwt_model_path': './models/fast-text/mwt/uk_iu_mwt_expander.pt'
    }

    config_glove = {
        'processors': 'pos, lemma, tokenize, depparse',
        'lang': 'uk',
        'depparse_model_path': './models/glove/depparse/uk_iu_parser.pt',
        'pos_pretrain_path': 'ewt_glove.pt',
        'depparse_pretrain_path': 'ewt_glove.pt',
        'tokenize_model_path': './models/glove/tokenize/uk_iu_tokenizer.pt',
        'pos_model_path': './models/glove/pos/uk_iu_tagger.pt',
        'lemma_model_path': './models/glove/lemma/uk_iu_lemmatizer.pt',
        'mwt_model_path': './models/glove/mwt/uk_iu_mwt_expander.pt'
    }

    model_original = stanza.Pipeline(**config_original)
    model_fast_text = stanza.Pipeline(**config_fast_text)
    model_glove = stanza.Pipeline(**config_glove)

    connector_original = StanzaConnector(model=model_original)
    connector_fast_text = StanzaConnector(model=model_fast_text)
    connector_glove = StanzaConnector(model=model_glove)
    # connector_trankit = TrankitConnector()

    classifier = DependencyParsingClassifier([connector_glove])
    predictions = classifier.predict_full_text(full_text)
    classifier.write_to_conllu("ensemble.conllu")
