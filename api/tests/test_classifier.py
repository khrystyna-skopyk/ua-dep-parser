import pytest
import os
import stanza

from stanza.models.common.pretrain import Pretrain
from api.classifier import DependencyParsingClassifier
from api.configs import config_fast_text, config_glove, config_original
from api.connectors import StanzaConnector, TrankitConnector

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)


@pytest.mark.parametrize('name, text, expected', [
    ["TP", "Це мало мало значення.",
     [[{'upos': 'PUNCT', 'head': 3, 'deprel': 'punct', 'id': 1, 'lemma': '«', 'xpos': 'U', 'feats': 'PunctType=Quot',
        'misc': 'start_char=0|end_char=1', 'text': '«', 'uas_weight': 0, 'las_weight': 0},
       {'upos': 'ADJ', 'head': 3, 'deprel': 'amod', 'id': 2, 'lemma': 'український', 'xpos': 'Ao-fsns', 'feats':
           'Case=Nom|Gender=Fem|Number=Sing', 'misc': 'start_char=1|end_char=11', 'text': 'Українська',
        'uas_weight': 0, 'las_weight': 0},
       {'upos': 'NOUN', 'head': 0, 'deprel': 'root', 'id': 3, 'lemma': 'правда', 'xpos': 'Ncfsnn',
        'feats': 'Animacy=Inan|Case=Nom|Gender=Fem|Number=Sing', 'misc': 'start_char=12|end_char=18', 'text': 'правда',
        'uas_weight': 0, 'las_weight': 0},
       {'upos': 'PUNCT', 'head': 3, 'deprel': 'punct', 'id': 4, 'lemma': '»', 'xpos': 'U', 'feats': 'PunctType=Quot',
        'misc': 'start_char=18|end_char=19', 'text': '»', 'uas_weight': 0, 'las_weight': 0},
       {'upos': 'PUNCT', 'head': 12, 'deprel': 'punct', 'id': 5, 'lemma': '—', 'xpos': 'U', 'feats': 'PunctType=Dash',
        'misc': 'start_char=20|end_char=21', 'text': '—', 'uas_weight': 0, 'las_weight': 0},
       {'upos': 'ADJ', 'head': 12, 'deprel': 'amod', 'id': 6, 'lemma': 'український', 'xpos': 'Ao-nsns',
        'feats': 'Case=Nom|Gender=Neut|Number=Sing', 'misc': 'start_char=22|end_char=32', 'text': 'українське',
        'uas_weight': 0, 'las_weight': 0},
       {'upos': 'ADJ', 'head': 9, 'deprel': 'compound', 'id': 7, 'lemma': 'суспільний', 'xpos': 'A',
        'feats': 'Hyph=Yes', 'misc': 'start_char=33|end_char=42', 'text': 'суспільно', 'uas_weight': 0,
        'las_weight': 0},
       {'upos': 'PUNCT', 'head': 7, 'deprel': 'punct', 'id': 8, 'lemma': '-', 'xpos': 'U', 'feats': 'PunctType=Hyph',
        'misc': 'start_char=42|end_char=43', 'text': '-', 'uas_weight': 0, 'las_weight': 0},
       {'upos': 'ADJ', 'head': 12, 'deprel': 'amod', 'id': 9, 'lemma': 'політичний', 'xpos': 'Ao-nsns',
        'feats': 'Case=Nom|Gender=Neut|Number=Sing', 'misc': 'start_char=43|end_char=52', 'text': 'політичне',
        'uas_weight': 0, 'las_weight': 0},
       {'upos': 'NOUN', 'head': 12, 'deprel': 'compound', 'id': 10, 'lemma': 'інтернет', 'xpos': 'Ncmsnn',
        'feats': 'Animacy=Inan|Case=Nom|Gender=Masc|Number=Sing', 'misc': 'start_char=53|end_char=61',
        'text': 'інтернет', 'uas_weight': 0, 'las_weight': 0},
       {'upos': 'PUNCT', 'head': 10, 'deprel': 'punct', 'id': 11, 'lemma': '-', 'xpos': 'U', 'feats': 'PunctType=Hyph',
        'misc': 'start_char=61|end_char=62', 'text': '-', 'uas_weight': 0, 'las_weight': 0},
       {'upos': 'NOUN', 'head': 3, 'deprel': 'appos', 'id': 12, 'lemma': 'ЗМІ', 'xpos': 'Y',
        'feats': 'Abbr=Yes|Animacy=Inan|Case=Nom|Gender=Neut|Number=Sing|Uninflect=Yes',
        'misc': 'start_char=62|end_char=65', 'text': 'ЗМІ', 'uas_weight': 0, 'las_weight': 0},
       {'upos': 'PUNCT', 'head': 15, 'deprel': 'punct', 'id': 13, 'lemma': ',', 'xpos': 'U', 'feats': '',
        'misc': 'start_char=65|end_char=66', 'text': ',', 'uas_weight': 0, 'las_weight': 0},
       {'upos': 'DET', 'head': 15, 'deprel': 'obj', 'id': 14, 'lemma': 'який', 'xpos': 'Pr--n-saa',
        'feats': 'Case=Acc|Gender=Neut|Number=Sing|PronType=Rel', 'misc': 'start_char=67|end_char=70', 'text': 'яке',
        'uas_weight': 0, 'las_weight': 0},
       {'upos': 'VERB', 'head': 12, 'deprel': 'acl:relcl', 'id': 15, 'lemma': 'заснувати', 'xpos': 'Vmeis-sm',
        'feats': 'Aspect=Perf|Gender=Masc|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin',
        'misc': 'start_char=71|end_char=79', 'text': 'заснував', 'uas_weight': 0, 'las_weight': 0},
       {'upos': 'PROPN', 'head': 15, 'deprel': 'nsubj', 'id': 16, 'lemma': 'Георгій', 'xpos': 'Npmsny',
        'feats': 'Animacy=Anim|Case=Nom|Gender=Masc|NameType=Giv|Number=Sing', 'misc': 'start_char=80|end_char=87',
        'text': 'Георгій', 'uas_weight': 0, 'las_weight': 0},
       {'upos': 'PROPN', 'head': 16, 'deprel': 'flat:name', 'id': 17, 'lemma': 'Гонгадзе', 'xpos': 'Npmsny',
        'feats': 'Animacy=Anim|Case=Nom|Gender=Masc|NameType=Sur|Number=Sing', 'misc': 'start_char=88|end_char=96',
        'text': 'Гонгадзе', 'uas_weight': 0, 'las_weight': 0},
       {'upos': 'ADP', 'head': 19, 'deprel': 'case', 'id': 18, 'lemma': 'у', 'xpos': 'Spsl', 'feats': 'Case=Loc',
        'misc': 'start_char=97|end_char=98', 'text': 'у', 'uas_weight': 0, 'las_weight': 0},
       {'upos': 'NOUN', 'head': 15, 'deprel': 'obl', 'id': 19, 'lemma': 'квітень', 'xpos': 'Ncmsln',
        'feats': 'Animacy=Inan|Case=Loc|Gender=Masc|Number=Sing', 'misc': 'start_char=99|end_char=105',
        'text': 'квітні', 'uas_weight': 0, 'las_weight': 0},
       {'upos': 'ADJ', 'head': 21, 'deprel': 'amod', 'id': 20, 'lemma': '2000', 'xpos': 'Mlomsg',
        'feats': 'Case=Gen|Gender=Masc|NumType=Ord|Number=Sing|Uninflect=Yes', 'misc': 'start_char=106|end_char=110',
        'text': '2000', 'uas_weight': 0, 'las_weight': 0},
       {'upos': 'NOUN', 'head': 19, 'deprel': 'nmod', 'id': 21, 'lemma': 'рік', 'xpos': 'Ncmsgn',
        'feats': 'Animacy=Inan|Case=Gen|Gender=Masc|Number=Sing', 'misc': 'start_char=111|end_char=115', 'text': 'року',
        'uas_weight': 0, 'las_weight': 0},
       {'upos': 'PUNCT', 'head': 3, 'deprel': 'punct', 'id': 22, 'lemma': '.', 'xpos': 'U', 'feats': '',
        'misc': 'start_char=115|end_char=116', 'text': '.', 'uas_weight': 0, 'las_weight': 0}]]],
    ["TP", "«Українська правда» — українське суспільно-політичне інтернет-ЗМІ, яке заснував Георгій Гонгадзе у квітні"
           " 2000 року.",
     [[{'upos': 'PRON', 'head': 4, 'deprel': 'nsubj', 'id': 1, 'lemma': 'це', 'xpos': 'Pd--nnsnn',
        'feats': 'Animacy=Inan|Case=Nom|Gender=Neut|Number=Sing|PronType=Dem', 'misc': 'start_char=0|end_char=2',
        'text': 'Це', 'uas_weight': 0, 'las_weight': 0},
       {'upos': 'ADV', 'head': 3, 'deprel': 'advmod', 'id': 2, 'lemma': 'мало', 'xpos': 'Rp', 'feats': 'Degree=Pos',
        'misc': 'start_char=3|end_char=7', 'text': 'мало', 'uas_weight': 0, 'las_weight': 0},
       {'upos': 'VERB', 'head': 4, 'deprel': 'root', 'id': 3, 'lemma': 'мало', 'xpos': 'Rp', 'feats': 'Degree=Pos',
        'misc': 'start_char=8|end_char=12', 'text': 'мало', 'uas_weight': 0, 'las_weight': 0},
       {'upos': 'NOUN', 'head': 0, 'deprel': 'root', 'id': 4, 'lemma': 'значення', 'xpos': 'Ncnsnn',
        'feats': 'Animacy=Inan|Case=Nom|Gender=Neut|Number=Sing', 'misc': 'start_char=13|end_char=21',
        'text': 'значення', 'uas_weight': 0, 'las_weight': 0},
       {'upos': 'PUNCT', 'head': 4, 'deprel': 'punct', 'id': 5, 'lemma': '.', 'xpos': 'U', 'feats': '',
        'misc': 'start_char=21|end_char=22', 'text': '.', 'uas_weight': 0, 'las_weight': 0}]]
     ]
])
def test_classifier(name, text, expected):
    pt_original = Pretrain(f"{ROOT_DIR}/ewt_original.pt", f"{ROOT_DIR}/models/original/ukoriginalvectors.xz")
    pt_fast_text = Pretrain(f"{ROOT_DIR}/ewt_fast_text.pt", f"{ROOT_DIR}/models/fast-text/uk.vectors.xz")
    pt_glove = Pretrain(f"{ROOT_DIR}/ewt_glove.pt", f"{ROOT_DIR}/models/glove/glove.xz")

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
    classifier = DependencyParsingClassifier(
        [connector_original, connector_fast_text, connector_glove, connector_trankit])
    predicted = classifier.predict_full_text(text)

    assert predicted == expected

