#download train scripts
git clone https://github.com/stanfordnlp/stanza-train.git
cd stanza-train
pip3 install -r requirements.txt

#download original stanza
git clone https://github.com/stanfordnlp/stanza.git
cp config/config.sh stanza/scripts/config.sh
cp config/xpos_vocab_factory.py stanza/stanza/models/pos/xpos_vocab_factory.py
cd stanza

#prepare data for training
wget https://github.com/olexandryermilov/ukrdata/raw/master/extern_data.zip
wget https://github.com/olexandryermilov/ukrdata/raw/master/data.zip
unzip extern_data.zip
unzip data.zip
rm extern_data.zip
rm data.zip

#train processors
python3 stanza/utils/datasets/prepare_pos_treebank.py UD_Ukrainian-IU
python3 stanza/utils/training/run_pos.py UD_Ukrainian-IU

python3 stanza/utils/datasets/prepare_tokenizer_treebank.py UD_Ukrainian-IU
python3 stanza/utils/training/run_tokenizer.py UD_Ukrainian-IU

python3 stanza/utils/datasets/prepare_mwt_treebank.py UD_Ukrainian-IU
python3 stanza/utils/training/run_mwt.py UD_Ukrainian-IU  

python3 stanza/utils/datasets/prepare_lemma_treebank.py UD_Ukrainian-IU 
python3 stanza/utils/training/run_lemma.py UD_Ukrainian-IU

python3 stanza/utils/datasets/prepare_depparse_treebank.py UD_Ukrainian-IU --gold
python3 stanza/utils/training/run_depparse.py UD_Ukrainian-IU 
