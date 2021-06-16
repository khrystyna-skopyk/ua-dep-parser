import os.path
import stanza
import shutil
import os

from api.data_loader import DataLoader
from trankit import Pipeline
from stanza.models.common.pretrain import Pretrain

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)


class PretrainInitializer:

    def __init__(self):
        self.fast_text_file = f'{ROOT_DIR}/ewt_fast_text.pt'
        self.original_file = f'{ROOT_DIR}/ewt_original.pt'
        self.glove_file = f'{ROOT_DIR}/ewt_glove.pt'

    def __init_stanza(self):
        

        fast_text_file_exists = os.path.isfile(self.fast_text_file) 
        glove_file_exists = os.path.isfile(self.glove_file) 
        original_file_exists = os.path.isfile(self.original_file) 

        if fast_text_file_exists and glove_file_exists and original_file_exists:
            return

        pt_original = Pretrain(self.original_file, f"{ROOT_DIR}/models/original/ukoriginalvectors.xz")
        pt_fast_text = Pretrain(self.fast_text_file, f"{ROOT_DIR}/models/fast-text/uk.vectors.xz")
        pt_glove = Pretrain(self.glove_file, f"{ROOT_DIR}/models/glove/glove.xz")
 
        pt_original.load()
        pt_fast_text.load()
        pt_glove.load()

        stanza.download('uk')

    def __init_trankit(self):
        if os.path.isdir(f'{ROOT_DIR}/cache') == True:
            return
        print("Initializing Trankit models")
        Pipeline('ukrainian', cache_dir=f'{ROOT_DIR}/cache', gpu=False)

    def initialize(self):
        self.__init_stanza()
        self.__init_trankit()

    def reinitialize(self):
        
        if os.path.isdir(f'{ROOT_DIR}/cache'):
            try:
                shutil.rmtree(f'{ROOT_DIR}/cache')
            except:
                pass
        
        if os.path.isdir(f'{ROOT_DIR}/models'):
            try:
                shutil.rmtree(f'{ROOT_DIR}/models')
            except:
                pass

        if os.path.isfile(self.fast_text_file):
            os.remove(self.fast_text_file)
        
        if os.path.isfile(self.original_file):
            os.remove(self.original_file)
        
        if os.path.isfile(self.glove_file):
            os.remove(self.glove_file)

        data_loader = DataLoader()
        data_loader.init_data()
        self.initialize()

