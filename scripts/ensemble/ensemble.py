import operator
from stanza_connector import StanzaConnector

class DependencyParsingClassifier:
    def __init__(self, connectors) -> None:
        self.connectors = connectors

    def append(self, connector):
        if connector != None:
            self.connectors.append(connector)

    def revert_predictions(self, predictions):
        result = []
        for index, prediction in enumerate(predictions):
            if len(result) == 0:
                result = [[] for number in range(len(prediction))]
            for index, word in enumerate(prediction):
                result[index].append(word)
        return result

    def merge_predictions(self, predictions):
        words = []
        for word_list in predictions:
            deprels = {} 
            heads = {}
            ids = {}
            texts= {}
            uposes = {}
            for word in word_list:
                self.create_values_dict(word.deprel, deprels, word.las_weight)
                self.create_values_dict(word.head, heads, word.uas_weight)
                self.create_values_dict(word.id, ids, word.uas_weight)
                self.create_values_dict(word.text, texts, word.uas_weight)
                self.create_values_dict(word.upos, uposes, word.uas_weight)
            word = self.merge_word_values(deprels, heads, ids, texts, uposes)
            words.append(word)
        return words

    def merge_word_values(self, deprels, heads, ids, texts, uposes):
        id = self.merge_dict_values(ids)
        head = self.merge_dict_values(heads)
        deprel = self.merge_dict_values(deprels)
        text = self.merge_dict_values(texts)
        upos = self.merge_dict_values(uposes)
        return [id, head, deprel, text, upos]


    def merge_dict_values(self, input_dict):

        for item in input_dict:
            input_dict[item]["probability"] = input_dict[item]["weight"] / input_dict[item]["count"]
        return max(input_dict.items(), key=operator.itemgetter(1))[0]

    def create_values_dict(self, value, input_dict, weigth):
            
        if value not in input_dict:
            input_dict[value] = {}
            input_dict[value]["count"] = 0
            input_dict[value]["weight"] = 0

        input_dict[value]["count"] += 1
        input_dict[value]["weight"] += weigth

    def predict(self, text):
        predictions = []
        for connector in self.connectors:
            prediction = connector.predict(text)
            predictions.append(prediction)
        predictions = self.revert_predictions(predictions)
        words = self.merge_predictions(predictions)
        return words

    
    


if __name__ == "__main__":
    connector1 = StanzaConnector()
    connector2 = StanzaConnector()
    classifier = DependencyParsingClassifier([connector1,connector2])
    predictions = classifier.predict("Украї́нська пра́вда — українське суспільно-політичне інтернет-ЗМІ, засноване у квітні 2000 року.")
    print(predictions)