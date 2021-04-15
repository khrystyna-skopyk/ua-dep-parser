from scripts.data_augmentation.conll import CoNLL
from scripts.data_augmentation.ud_augment import DataAugmentation
from datetime import datetime
import os
import re

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_DIR = os.path.dirname(SCRIPTS_DIR)


def doc_to_text(doc):
    """

    :param doc:
    :return:
    """
    text = ""
    for token in doc:
        misc = token.get("misc", "")
        text += token["text"] if misc else ""
        searched = re.search("SpaceAfter=[^|]+", misc)
        if not searched or searched.group().split("=")[1] != "No":
            text += " "
    return text.strip()


def doc_to_translit(doc):
    """

    :param doc:
    :return:
    """
    translit = ""
    for token in doc:
        misc = token.get("misc", "")
        searched_translit = re.search("\\|Translit=.+$", misc)
        translit += searched_translit.group().split("=")[1] if searched_translit else ""
        searched_space = re.search("SpaceAfter=[^|]+", misc)
        if not searched_space or searched_space.group().split("=")[1] != "No":
            translit += " "
    return translit.strip()


def doc_to_lines(doc):
    """
    Creates necessary strings to dump them
    :param doc: list of dictionaries
    :return: list of strings
    """
    lines = []
    for d in doc:
        line = ""
        if d.get('id'):
            index = str(d['id'][0]) if len(d['id']) == 1 else f"{d['id'][0]}-{d['id'][1]}"
            line += index + "\t"
        else:
            line += "_\t"
        if d.get('text'):
            line += str(d['text']) + "\t"
        else:
            line += "_\t"
        if d.get('lemma'):
            line += str(d['lemma']) + "\t"
        else:
            line += "_\t"
        if d.get('upos'):
            line += str(d['upos']) + "\t"
        else:
            line += "_\t"
        if d.get('xpos'):
            line += str(d['xpos']) + "\t"
        else:
            line += "_\t"
        if d.get('feats'):
            line += str(d['feats']) + "\t"
        else:
            line += "_\t"
        if d.get('head'):
            line += str(d['head']) + "\t"
        else:
            line += "_\t"
        if d.get('deprel'):
            line += str(d['deprel']) + "\t"
        else:
            line += "_\t"
        if d.get('deps'):
            line += str(d['deps']) + "\t"
        else:
            line += "_\t"
        if d.get('misc'):
            line += str(d['misc'])
        else:
            line += "_\t"
        lines.append(line)
    return lines


def augment_file(fconllu, verbose=True):
    """
    Run and augmentation process on the file.
    :param fconllu: name of the file in conllu format
    :param verbose: boolean
    """
    doc_dct = CoNLL.conll2dict(input_file=fconllu)
    fname = fconllu.split("/")[-1].split(".")[0]
    outf = open(f"{MAIN_DIR}/data/augmented/{fname}-aug.conllu", "w+")
    textoutf = open(f"{MAIN_DIR}/data/augmented/{fname}-aug.txt", "w+")
    n_augmented = 0
    n_sents = 0
    for doc in doc_dct:
        doc_to_aug = DataAugmentation(doc)
        doc_to_aug.augment()
        aug_sents = doc_to_aug.get_aug_docs()
        for aug_s in aug_sents:
            text = f"# text = {doc_to_text(aug_s)}"
            translit = f"# translit = {doc_to_translit(aug_s)}"
            lines = doc_to_lines(aug_s)
            lines = '\n'.join(lines)
            outf.write(f"{text}\n{translit}\n{lines}\n\n")
            textoutf.write(f"{text[9:]}\n")
        n_augmented += doc_to_aug.get_n_aug()
        n_sents +=1
        if verbose:
            print(f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S")} '
                  f'INFO: {doc_to_aug} Overall augmented {n_augmented} sentence(s) from {n_sents} original.')
    outf.close()
    textoutf.close()


if __name__ == "__main__":
    #file_conllu = f"{MAIN_DIR}/data/UD_Ukrainian-IU/for-tests.conllu"
    file_conllu = f"{MAIN_DIR}/data/UD_Ukrainian-IU/uk_iu-ud-train.conllu"
    augment_file(file_conllu)


# TO FIX:
# У плані генеральному міста територія, на якій орендар – ТОВ “Київщина-Житло” – багатоповерхівки зводитиме, досі позначена як «ліси», цей план не роками змінювався.
# здобув 2007-го ступінь магістра наук політичних того самого вишу.
# Ода Київська виявилася чи не першою, яка намагалася вплинути на перевізників.
# А там, на землі, панує спокій ясний, мрія ворога мого цвіте і витає — і ніщо не її бентежить.
# Персики Найдешевші, які вдалося знайти на ринку Привокзальному у Львові, — по 12 гривень за кілограм, і то — молдовські.
# Книжка тимченкова, що в ній близько 70 сторінок відведено фонології, відзначалася викладом атомістичним, типовим для школи молодограматичної: зміни кожного звука представлено в ізоляції від інших; часові й територіальні характеристики не майже враховуються, спроби нечисленні аналізу хронологічного є непевними через брак підходу критичного до пам’яток матеріалу.
# Справа серпня 31 (заворушення Масові серпня 31, Справа Конституції захисників) — провадження кримінальне, яке розслідується поліцією Національною, безпеки Службою та Генеральною України прокуратурою за фактом заворушень масових, поєднаних із нанесенням ушкоджень тілесних міліції працівникам та військовослужбовцям Національної України гвардії, а також смертей та поранень міліції працівників та військовослужбовців Національної України гвардії в результаті метання, за версією влади української, гранати РГО з натовпу в правоохоронців лаву, що було кваліфіковано як організований акт терористичний.

