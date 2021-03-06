import operator
import stanza
import os

from stanza.models.common.pretrain import Pretrain
from api.configs import config_fast_text, config_glove, config_original
from api.models import Word, Sentence
from api.helpers import Graph
from api.connectors import StanzaConnector, TrankitConnector
from collections import OrderedDict
from conllu.models import TokenList

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)


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
                if len(batches) <= index_prediction:
                    continue
                if len(batches[index_prediction]) == 0:
                    batches[index_prediction] = [[] for number in range(len(prediction.words))]
                for index, word in enumerate(prediction.words):
                    if len(batches) <= index_prediction:
                        continue
                    if len(batches[index_prediction]) <= index:
                        continue
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

                word = self.merge_word_values(deprels, heads, ids, texts, uposes, lemmas, xposes, feats_list, miscs)
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

    def predict_full_text(self, text):
        sentences = text.split('\n')
        parsed_sentences = []
        for sentence in sentences:
            parsed_sentences += self.predict(sentence)
            print(sentence)
        self.sentences = parsed_sentences
        return parsed_sentences

    def calculate_soft_score(self, x_predictions, y_predictions):
        matches = 0
        x_words = []
        y_words = []
        for sentence in x_predictions:
            x_words += sentence.words
        for sentence in y_predictions:
            y_words += sentence.words

        for index, word in enumerate(x_words):
            if index >= len(y_words):
                continue 
            if x_words[index].deprel != y_words[index].deprel:
                continue
            if x_words[index].head != y_words[index].head:
                continue
            matches+=1
        return matches / len(x_predictions)

    def soft_voting(self, predictions):
        predictions_dict = {}
        for index, prediction in enumerate(predictions):
            predictions_dict[index] = prediction

        scores_dict = {}
        for prediction_i in predictions_dict:
            for prediction_j in predictions_dict:
                if prediction_i == prediction_j:
                    continue
                score = self.calculate_soft_score(predictions_dict[prediction_i],predictions_dict[prediction_j])
                if prediction_i not in scores_dict:
                    scores_dict[prediction_i] = 1
                scores_dict[prediction_i] *= score

        for scores_key in scores_dict:
            if len(predictions_dict[scores_key]) == 0:
                continue
            sentence = predictions_dict[scores_key][0]
            if len(sentence.words) == 0:
                continue
            las_weight = sentence.words[0].las_weight
            scores_dict[scores_key] *= las_weight
        selection_index = max(scores_dict.items(), key=operator.itemgetter(1))[0]
        return predictions_dict[selection_index]


    def predict(self, sentence):
        predictions = []
        for connector in self.connectors:
            prediction = connector.predict(sentence)
            predictions.append(prediction)
        reverted_predictions = self.revert_predictions(predictions)
        parsed_sentences = self.merge_predictions(reverted_predictions)
        circle_dependency = False
        for sentence in parsed_sentences:
            check_result = self.check_for_circle(sentence)
            if check_result == True:
                circle_dependency = True
        if circle_dependency == True:
            parsed_sentences = self.soft_voting(predictions)
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
                graph.append_edge(relation, connection)
        return graph.check_is_cyclic()


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
    
    # with open(f'{ROOT_DIR}/data/UD_Ukrainian-IU/uk_iu-ud-test.txt') as f:
    #     full_text = f.read()

    full_text = "?????????????? ???????????????????? ?????????????????? - ??????????????????, ????????????????, ??????????????, ????????????, - ???? ???????????????? ???? ?????????? ?????????????????????? ?? ?????????????????????? 45-???? ???????????????????????? ???????????????????????? ???????????? ???? ???????????? ?????????????? ???????????? ???????????????????????? ??????."
    
    pt_original = Pretrain(f"{ROOT_DIR}/models/original/ewt_original.pt", f"{ROOT_DIR}/models/original/ukoriginalvectors.xz")
    pt_fast_text = Pretrain(f"{ROOT_DIR}/models/fast-text/ewt_fast_text.pt", f"{ROOT_DIR}/models/fast-text/uk.vectors.xz")
    pt_glove = Pretrain(f"{ROOT_DIR}/models/glove/ewt_glove.pt", f"{ROOT_DIR}/models/glove/glove.xz")
 
    pt_original.load()
    pt_fast_text.load()
    pt_glove.load()

    model_original = stanza.Pipeline(**config_original)
    model_fast_text = stanza.Pipeline(**config_fast_text)
    model_glove = stanza.Pipeline(**config_glove)

    connector_original = StanzaConnector(model=model_original)
    connector_fast_text = StanzaConnector(model=model_fast_text)
    connector_glove = StanzaConnector(model=model_glove)
    connector_trankit = TrankitConnector()

    classifier = DependencyParsingClassifier([connector_original, connector_fast_text, connector_glove, connector_trankit])
    predictions = classifier.predict_full_text(full_text)
    classifier.write_to_conllu(f"{ROOT_DIR}/data/conllu-generated/ensemble.conllu")
