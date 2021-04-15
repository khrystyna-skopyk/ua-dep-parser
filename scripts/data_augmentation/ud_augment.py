# TODO:
# - split change_shuffled into several funcs

from collections import Counter, defaultdict
from copy import deepcopy
import re


class DataAugmentation(object):

    def __init__(self, sent_doc):
        """
        Creates an instances of a document (sentence) for augmentation.
        :param sent_doc: list of dictionaries (UD type)
        :param deps: list of tuples, dependencies between tokens
        :param augmented: list of augmented docs
        :param n_aug: integer, number of augmented sentences
        """
        self.doc = sent_doc
        self.has_mwt = any(True for el in self.doc if len(el['id']) == 2)
        self.deps = [(el["id"][0], el["deps"]) for el in self.doc if not self.has_mwt]
        self.augmented = []
        self.n_aug = 0

    def has_deeper_dep(self, index):
        """
        Checks if a token has dependents.
        :param index: integer, index of a token
        :return: boolean
        """
        for _, head_dep in self.deps:
            head_lst = re.findall("\\d+", head_dep)
            if f"{index}" in head_lst:
                return True
        return False

    def find_deps_to_aug(self):
        """
        Checks for tokens which can be shuffled.
        :return: list of tuples; a tuple contains indices for shuffling, a head index and child index
        """
        to_aug = []
        for child_id, head_dep in self.deps:
            if "|" in head_dep:
                continue
            matched = re.match("\\d+", head_dep)
            head_id = int(matched.group())
            dep = head_dep[matched.end()+1:]
            if abs(child_id-head_id) == 1 and not self.has_deeper_dep(child_id):
                child_dct = self.doc[child_id - 1]
                if dep in ['nmod', 'amod', 'obj', 'xcomp', 'obl']:
                    to_aug.append((head_id, child_id))
                elif dep in ['nsubj'] and self.doc[head_id-1]['upos'] == 'VERB':
                    to_aug.append((head_id, child_id))
                elif dep in ['advmod'] and not self.doc[head_id-1]['upos'] in ['ADV', 'ADJ']:
                    adv_feats = child_dct.get('feats', "")
                    if "Polarity=Neg" != adv_feats and "PronType=Rel" != adv_feats and "PronType=Int" != adv_feats:
                        to_aug.append((head_id, child_id))
        return to_aug

    @staticmethod
    def change_shuffled(new_lst, head_child_to_aug):
        """
        Changes all necessary indices and misc for each word in the document.
        :param new_lst: list of dictionaries, a dictionary contains all info on the word (including a dependency)
        :param head_child_to_aug: list of tuples; a tuple contains a head index and a child index
        :return: updated list of dictionaries
        """
        dct_of_changes = dict(head_child_to_aug)
        heads = dct_of_changes.keys()
        for el in new_lst:
            if el.get('head') and el['head'] in heads:
                el['head'] = dct_of_changes[el['head']]
                el['deps'] = f"{el['head']}:{el['deprel']}"
        for new_c, new_h in head_child_to_aug:
            ### capital letter
            if new_c in [1, 2] or new_h in [1, 2]:
                old_first_misc = new_lst[max(new_c,new_h)-1]['misc']
                should_upper_first = old_first_misc.split("|")[-1].split("=")[1][0].isupper() #
                old_second_text = new_lst[min(new_c,new_h)-1]['text']
                should_upper_second = old_second_text[0].isupper() #
                if should_upper_first:
                    misc = new_lst[min(new_c,new_h)-1]['misc'].split("|")
                    new_lst[min(new_c,new_h)-1]['misc'] = \
                        f"{'|'.join(misc[:-1])}|Translit={misc[-1].split('=')[1].capitalize()}"
                    new_lst[min(new_c,new_h)-1]['text'] = new_lst[min(new_c,new_h)-1]['text'].capitalize()
                if not should_upper_second:
                    misc = new_lst[max(new_c,new_h)-1]['misc'].split("|")
                    new_lst[max(new_c,new_h)-1]['misc'] = \
                        f"{'|'.join(misc[:-1])}|Translit={misc[-1].split('=')[1].lower()}"
                    new_lst[max(new_c,new_h)-1]['text'] = new_lst[max(new_c,new_h)-1]['text'].lower()
            ### spacing
            misc_new_c = new_lst[new_c-1]['misc'].split("|")
            misc_new_h = new_lst[new_h-1]['misc'].split("|")
            if 'SpaceAfter=No' in misc_new_c and 'SpaceAfter=No' in misc_new_h:
                continue
            elif 'SpaceAfter=No' in misc_new_c:
                misc_new_c = ("|").join(misc_new_c[:-2] + [misc_new_c[-1]])
                misc_new_h = ("|").join(misc_new_h[:-1] + ['SpaceAfter=No'] + [misc_new_h[-1]])
                new_lst[new_c-1]['misc'] = misc_new_c
                new_lst[new_h-1]['misc'] = misc_new_h
            elif 'SpaceAfter=No' in misc_new_h:
                misc_new_c = ("|").join(misc_new_c[:-1] + ['SpaceAfter=No'] + [misc_new_c[-1]])
                misc_new_h = ("|").join(misc_new_h[:-2] + [misc_new_h[-1]])
                new_lst[new_c-1]['misc'] = misc_new_c
                new_lst[new_h-1]['misc'] = misc_new_h
        return new_lst

    def shuffle(self, head_child_to_aug):
        """
        Shuffles the original document and creates an augmented one.
        :param head_child_to_aug: list of tuples; a tuple contains a head index and a child index
        """
        new_lst = []
        previd = 0
        doc_copy = deepcopy(self.doc)
        for h, c in head_child_to_aug:
            minid, maxid = min(c, h), max(c, h)
            firstel = doc_copy[maxid-1]
            firstel['id'] = (minid,)
            lastel = doc_copy[minid-1]
            lastel['id'] = (maxid,)
            new_lst += doc_copy[previd:minid-1] + [firstel] + [lastel]
            previd = maxid
        if previd != len(doc_copy):
            new_lst += doc_copy[previd:]
        changed_lst = self.change_shuffled(new_lst, head_child_to_aug)
        self.augmented.append(changed_lst)
        self.n_aug += 1

    def rec(self, head_child_to_aug, id_dct):
        """
        A recursive func to create different combinations (lists) of indices for augmentation.
        :param head_child_to_aug: list of tuples; a tuple contains a head index and a child index
        :param id_dct: dictionary of head indices and their places
        """
        id_dct_copy = deepcopy(id_dct)
        if id_dct_copy:
            k = list(id_dct_copy.keys())[0]
            first_to_aug = deepcopy(head_child_to_aug)
            del first_to_aug[first_to_aug.index((k, id_dct_copy[k][0]))]
            second_to_aug = deepcopy(head_child_to_aug)
            del second_to_aug[second_to_aug.index((k, id_dct_copy[k][1]))]
            del id_dct_copy[k]
            self.rec(first_to_aug, id_dct_copy)
            self.rec(second_to_aug, id_dct_copy)
        else:
            self.shuffle(head_child_to_aug)

    def split_augmentation(self, head_child_to_aug):
        """
        Checks if one head has two dependents for augmentation.
        If yes, calls a recursive func to create different combinations of indices for augmentation.
        :param head_child_to_aug: list of tuples; a tuple contains a head index and a child index
        """
        head_freq = Counter(el[0] for el in head_child_to_aug)
        head_indices = defaultdict(list)
        for h, c in head_child_to_aug:
            if head_freq[h] == 2:
                head_indices[h].append(c)
        if head_indices:
            self.rec(head_child_to_aug, head_indices)
        else:
            self.shuffle(head_child_to_aug)

    def augment(self):
        """
        Starts the process for document augmentation.
        """
        if not self.has_mwt:
            deps_to_aug = self.find_deps_to_aug()
            if deps_to_aug:
                self.split_augmentation(deps_to_aug)

    def __str__(self):
        """
        Prints info about the augmented sentences for the document.
        """
        return f"Created {self.n_aug} augmented sentence(s)."

    def get_n_aug(self):
        """
        Gets the number of augmented sentences.
        :return: integer
        """
        return self.n_aug

    def get_aug_docs(self):
        """
        Gets the list of augmented sentences.
        :return: list of lists (augmented sentences)
        """
        return self.augmented
