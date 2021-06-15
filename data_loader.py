import wget
import os
import zipfile
import shutil

class DataLoader:
    original_ud = 'https://github.com/olexandryermilov/ukrdata/raw/master/original_embedding_original_ud.zip'
    fast_text = 'https://github.com/olexandryermilov/ukrdata/raw/master/fasttext_embedding_augmented_ud.zip'
    glove = 'https://github.com/olexandryermilov/ukrdata/raw/master/glove_embedding_augmented_ud.zip'

    def __load(self):
        print('Beginning to download models')
        os.mkdir("models")
        print("Downloading original_ud")
        wget.download(self.original_ud, './models/')
        print("\nDownloading fast_text")
        wget.download(self.fast_text, './models/')
        print("\nDownloading glove")
        wget.download(self.glove, './models/')

        print("\nModels has been downloaded ")
        

    def __extract(self):

        print("\nExtracting data")
        try:
            with zipfile.ZipFile('./models/original_embedding_original_ud.zip', 'r') as zip_ref:
                zip_ref.extractall('./models/')
            with zipfile.ZipFile('./models/fasttext_embedding_augmented_ud.zip', 'r') as zip_ref:
                zip_ref.extractall('./models/')
            with zipfile.ZipFile('./models/glove_embedding_augmented_ud.zip', 'r') as zip_ref:
                zip_ref.extractall('./models/')
        except:
            pass

        print('\nRemoving unnecessary files')
        try:
            shutil.rmtree('./models/__MACOSX')
        except:
            pass

        try:
            os.remove('./models/original_embedding_original_ud.zip')
            os.remove('./models/fasttext_embedding_augmented_ud.zip')
            os.remove('./models/glove_embedding_augmented_ud.zip')
        except:
            pass
        
        try:
            os.rename('./models/glove_embedding_augmented_ud', './models/glove')
            os.rename('./models/model', './models/fast-text')
            os.rename('./models/original_embedding_original_ud', './models/original')
        except:
            pass

    def init_data(self):
        
        if os.path.isdir('models') == True:
            return

        self.__load()
        self.__extract()



if __name__ == "__main__":
    data_loader = DataLoader()
    data_loader.init_data()
    

