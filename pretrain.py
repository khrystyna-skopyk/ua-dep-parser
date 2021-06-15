import os.path
import stanza
import shutil

from data_loader import DataLoader
from trankit import Pipeline, pipeline
from stanza.models.common.pretrain import Pretrain

class PretrainInitializer:

    def __init__(self):
        self.fast_text_file = 'ewt_fast_text.pt'
        self.original_file = 'ewt_original.pt'
        self.glove_file = 'ewt_glove.pt'

    def __init_stanza(self):
        

        fast_text_file_exists = os.path.isfile(self.fast_text_file) 
        glove_file_exists = os.path.isfile(self.glove_file) 
        original_file_exists = os.path.isfile(self.original_file) 

        if fast_text_file_exists and glove_file_exists and original_file_exists:
            return

        pt_original = Pretrain(self.original_file, "./models/original/ukoriginalvectors.xz")
        pt_fast_text = Pretrain(self.fast_text_file, "./models/fast-text/uk.vectors.xz")
        pt_glove = Pretrain(self.glove_file, "./models/glove/glove.xz")
 
        pt_original.load()
        pt_fast_text.load()
        pt_glove.load()

        stanza.download('uk')

    def __init_trankit(self):
        if os.path.isdir('cache') == True:
            return
        print("Initializing Trankit models")
        Pipeline('ukrainian', cache_dir='./cache', gpu=False)

    def initialize(self):
        self.__init_stanza()
        self.__init_trankit()

    def reinitialize(self):
        
        if os.path.isdir('cache'):
            try:
                shutil.rmtree("cache")
            except:
                pass
        
        if os.path.isdir('models'):
            try:
                shutil.rmtree("models")
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


        
