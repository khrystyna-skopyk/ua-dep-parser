import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

config_original = {
        'processors': 'pos, lemma, tokenize, depparse',
        'lang': 'uk',
        'depparse_model_path': f'{ROOT_DIR}/models/original/depparse/uk_iu_parser.pt',
        'pos_pretrain_path': f'{ROOT_DIR}/ewt_original.pt',
        'depparse_pretrain_path': f'{ROOT_DIR}/ewt_original.pt',
        'tokenize_model_path': f'{ROOT_DIR}/models/original/tokenize/uk_iu_tokenizer.pt',
        'pos_model_path': f'{ROOT_DIR}/models/original/pos/uk_iu_tagger.pt',
        'lemma_model_path': f'{ROOT_DIR}/models/original/lemma/uk_iu_lemmatizer.pt',
        'mwt_model_path': f'{ROOT_DIR}/models/original/mwt/uk_iu_mwt_expander.pt'
    }

config_fast_text = {
        'processors': 'pos, lemma, tokenize, depparse',
        'lang': 'uk',
        'depparse_model_path': f'{ROOT_DIR}/models/fast-text/depparse/uk_iu_parser.pt',
        'pos_pretrain_path': f'{ROOT_DIR}/ewt_fast_text.pt',
        'depparse_pretrain_path': f'{ROOT_DIR}/ewt_fast_text.pt',
        'tokenize_model_path': f'{ROOT_DIR}/models/fast-text/tokenize/uk_iu_tokenizer.pt',
        'pos_model_path': f'{ROOT_DIR}/models/fast-text/pos/uk_iu_tagger.pt',
        'lemma_model_path': f'{ROOT_DIR}/models/fast-text/lemma/uk_iu_lemmatizer.pt',
        'mwt_model_path': f'{ROOT_DIR}/models/fast-text/mwt/uk_iu_mwt_expander.pt'
    }

config_glove = {
        'processors': 'pos, lemma, tokenize, depparse',
        'lang': 'uk',
        'depparse_model_path': f'{ROOT_DIR}/models/glove/depparse/uk_iu_parser.pt',
        'pos_pretrain_path': f'{ROOT_DIR}/ewt_glove.pt',
        'depparse_pretrain_path': f'{ROOT_DIR}/ewt_glove.pt',
        'tokenize_model_path': f'{ROOT_DIR}/models/glove/tokenize/uk_iu_tokenizer.pt',
        'pos_model_path': f'{ROOT_DIR}/models/glove/pos/uk_iu_tagger.pt',
        'lemma_model_path': f'{ROOT_DIR}/models/glove/lemma/uk_iu_lemmatizer.pt',
        'mwt_model_path': f'{ROOT_DIR}/models/glove/mwt/uk_iu_mwt_expander.pt'
    }

