from connectors import StanzaConnector
from flask import Flask
from flask import request
import stanza

from data_loader import DataLoader
from configs import config_original, config_fast_text, config_glove
from stanza.models.common.pretrain import Pretrain
from classifier import DependencyParsingClassifier


app = Flask(__name__)

@app.route("/check", methods=["GET"])
def check():
    return str("success")

@app.route("/", methods=["GET","POST"])
def parse_text():
    if request.method == "GET":
        return ""
    text = request.form["text"]

    pt_original = Pretrain("ewt_original.pt", "./models/original/ukoriginalvectors.xz")
    pt_fast_text = Pretrain("ewt_fast_text.pt", "./models/fast-text/uk.vectors.xz")
    pt_glove = Pretrain("ewt_glove.pt", "./models/glove/glove.xz")
 
    pt_original.load()
    pt_fast_text.load()
    pt_glove.load()

    model_original = stanza.Pipeline(**config_original)
    model_fast_text = stanza.Pipeline(**config_fast_text)
    model_glove = stanza.Pipeline(**config_glove)

    connector_original = StanzaConnector(model=model_original)
    connector_fast_text = StanzaConnector(model=model_fast_text)
    connector_glove = StanzaConnector(model=model_glove)

    classifier = DependencyParsingClassifier([connector_original, connector_fast_text, connector_glove])
    predictions = classifier.predict_full_text(text)
    response = prepare_response(predictions)
    return str(response)

def prepare_response(predictions):
    result = []
    for prediction in predictions:
        sentence = []
        for word in prediction.words:
            sentence.append(word.__dict__)
        result.append(sentence)
    return result


if __name__ == "__main__":
    data_loader = DataLoader()
    data_loader.init_data()
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)