from connectors import StanzaConnector, TrankitConnector
from flask import Flask
from flask import request, render_template
from flask_bootstrap import Bootstrap
import stanza

from data_loader import DataLoader
from configs import config_original, config_fast_text, config_glove
from classifier import DependencyParsingClassifier
from pretrain import PretrainInitializer


app = Flask(__name__)
bootstrap = Bootstrap(app)

@app.route("/check", methods=["GET"])
def check():
    return str("success")

@app.route("/reinitialize-pretrain", methods=["GET"])
def reinitialize_pretrain():
    pretrain_initializer = PretrainInitializer()
    pretrain_initializer.reinitialize()
    return str("success")

@app.route("/", methods=["GET","POST"])
def parse_text():
    if request.method == "GET":
        return render_template('/index.html')
    text = request.form["text"]

    model_original = stanza.Pipeline(**config_original, use_gpu=False)
    model_fast_text = stanza.Pipeline(**config_fast_text, use_gpu=False)
    model_glove = stanza.Pipeline(**config_glove, use_gpu=False)

    connector_trankit = TrankitConnector()
    connector_original = StanzaConnector(model=model_original)
    connector_fast_text = StanzaConnector(model=model_fast_text)
    connector_glove = StanzaConnector(model=model_glove)

    classifier = DependencyParsingClassifier([connector_original, connector_fast_text, connector_glove, connector_trankit])
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
    pretrain_initializer = PretrainInitializer()
    data_loader.init_data()
    pretrain_initializer.initialize()
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)