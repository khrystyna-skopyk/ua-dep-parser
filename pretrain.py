import os.path
import stanza
from trankit import Pipeline, pipeline
from stanza.models.common.pretrain import Pretrain

class PretrainInitializer:
    def __init_stanza(self):
        fast_text_file = 'ewt_fast_text.pt'
        original_file = 'ewt_original.pt'
        glove_file = 'ewt_glove.pt'

        fast_text_file_exists = os.path.isfile(fast_text_file) 
        glove_file_exists = os.path.isfile(glove_file) 
        original_file_exists = os.path.isfile(original_file) 

        if fast_text_file_exists and glove_file_exists and original_file_exists:
            return

        pt_original = Pretrain(original_file, "./models/original/ukoriginalvectors.xz")
        pt_fast_text = Pretrain(fast_text_file, "./models/fast-text/uk.vectors.xz")
        pt_glove = Pretrain(glove_file, "./models/glove/glove.xz")
 
        pt_original.load()
        pt_fast_text.load()
        pt_glove.load()

        stanza.download('uk')

    def __init_trankit(self):
        if os.path.isdir('cache') == True:
            return
        print("Initializing Trankit models")
        Pipeline('ukrainian', cache_dir='./cache')

    def initialize(self):
        self.__init_stanza()
        self.__init_trankit()