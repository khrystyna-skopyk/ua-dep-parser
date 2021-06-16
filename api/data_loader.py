import wget
import os
import zipfile
import shutil

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)


class DataLoader:
    original_ud = 'https://github.com/olexandryermilov/ukrdata/raw/master/original_embedding_original_ud.zip'
    fast_text = 'https://github.com/olexandryermilov/ukrdata/raw/master/fasttext_embedding_augmented_ud.zip'
    glove = 'https://github.com/olexandryermilov/ukrdata/raw/master/glove_embedding_augmented_ud.zip'

    def __load(self):
        print('Beginning to download models')
        os.mkdir(f'{ROOT_DIR}/models')
        print("Downloading original_ud")
        wget.download(self.original_ud, f'{ROOT_DIR}/models/')
        print("\nDownloading fast_text")
        wget.download(self.fast_text, f'{ROOT_DIR}/models/')
        print("\nDownloading glove")
        wget.download(self.glove, f'{ROOT_DIR}/models/')

        print("\nModels have been downloaded.")
        

    def __extract(self):

        print("\nExtracting data")
        try:
            with zipfile.ZipFile(f'{ROOT_DIR}/models/original_embedding_original_ud.zip', 'r') as zip_ref:
                zip_ref.extractall(f'{ROOT_DIR}/models/')
            with zipfile.ZipFile(f'{ROOT_DIR}/models/fasttext_embedding_augmented_ud.zip', 'r') as zip_ref:
                zip_ref.extractall(f'{ROOT_DIR}/models/')
            with zipfile.ZipFile(f'{ROOT_DIR}/models/glove_embedding_augmented_ud.zip', 'r') as zip_ref:
                zip_ref.extractall(f'{ROOT_DIR}/models/')
        except:
            pass

        print('\nRemoving unnecessary files')
        try:
            shutil.rmtree(f'{ROOT_DIR}/models/__MACOSX')
        except:
            pass

        try:
            os.remove(f'{ROOT_DIR}/models/original_embedding_original_ud.zip')
            os.remove(f'{ROOT_DIR}/models/fasttext_embedding_augmented_ud.zip')
            os.remove(f'{ROOT_DIR}/models/glove_embedding_augmented_ud.zip')
        except:
            pass
        
        try:
            os.rename(f'{ROOT_DIR}/models/glove_embedding_augmented_ud', f'{ROOT_DIR}/models/glove')
            os.rename(f'{ROOT_DIR}/models/model', f'{ROOT_DIR}/models/fast-text')
            os.rename(f'{ROOT_DIR}/models/original_embedding_original_ud', f'{ROOT_DIR}/models/original')
        except:
            pass

    def init_data(self):
        
        if os.path.isdir(f'{ROOT_DIR}/models'):
            return

        self.__load()
        self.__extract()


if __name__ == "__main__":
    data_loader = DataLoader()
    data_loader.init_data()

