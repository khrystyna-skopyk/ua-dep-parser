from stanza_connector import StanzaConnector

class DependencyParsingClassifier:
    def __init__(self, connectors) -> None:
        self.connectors = connectors

    def append(self, connector):
        if connector != None:
            self.connectors.append(connector)

    def predict(self, text):
        predictions = []
        for connector in self.connectors:
            prediction = connector.predict(text)
            predictions.append([prediction,connector.uas_weight, connector.las_weight])

        merged_predictions = self.__merge_predictions(predictions)
        print(merged_predictions)

    def __merge_predictions(self, predictions):
        merged_predictions = {}
        for prediction in predictions:
            for prediction_index in prediction[0].keys():
                if prediction_index not in merged_predictions:
                    merged_predictions[prediction_index] = []
                id = prediction[0][prediction_index]['id']
                upos = prediction[0][prediction_index]['upos']
                head = prediction[0][prediction_index]['head']
                uas = prediction[1]
                las = prediction[2]
                merged_predictions[prediction_index].append([(id, upos, head),uas, las])
        return merged_predictions

    def __process_predictions(self, merged_predictions):
        pass


if __name__ == "__main__":
    connector1 = StanzaConnector()
    connector2 = StanzaConnector()
    classifier = DependencyParsingClassifier([connector1,connector2])
    classifier.predict("Украї́нська пра́вда — українське суспільно-політичне інтернет-ЗМІ, засноване у квітні 2000 року.")
