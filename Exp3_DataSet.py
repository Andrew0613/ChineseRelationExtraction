import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json
from Exp3_Config import Training_Config
from torch import nn
# 训练集和验证集
class BasicDataset(Dataset):
    def __init__(self,rel2id, char2id, config):
        """
        input:
        rel2id: a dict, key: relation name, value: relation id
        char2id: a dict, key: char, value: char id
        config: a config class
        """
        super().__init__()
        self.rel2id = rel2id
        self.char2id = char2id
        self.config = config
        # Generate position embeddings
    def get_pos_embeded(self,i,lf1,rg1,lf2,rg2,maxlen=80):
        """
        input:
        i: the index of the char in the sentence
        lf1: the left-most position of the head entity
        rg1: the right-most position of the head entity
        lf2: the left-most position of the tail entity
        rg2: the right-most position of the tail entity
        maxlen: the max length of the sentence
        
        return:
        pos1_code: the relative position of the head entity
        pos2_code: the relative position of the tail entity
        """

        # scale to [0,2*max-length+2]
        def pos_embed(x):
            if x < -1*maxlen:
                return 0
            if -1*maxlen <= x <= maxlen:
                return x + maxlen + 1
            if x > maxlen:
                return 2*maxlen+2
        # corresponding to Eq. 1 in paper      
        def pos_embed2(i,l,r):
            if i>=l and i<=r:
                x = 0
            elif i<l:
                x = i-l
            else:
                x = i-r
            return pos_embed(x)
                                    
        return pos_embed2(i,lf1,rg1),pos_embed2(i,lf2,rg2)
    def position_padding(self, position:list):
        """
        input:
        position: a list of relative position
        return:
        position: a padded list of position embedding"""
        if len(position)<self.max_len:
            position.extend([2*self.max_len+2]*(self.max_len-len(position)))
        else:
            position = position[:self.max_len]
        return position
    def add_position(self,sentence,pos):
        """
        input:
        sentence: a str, the sentence
        pos: a dict, the position of the head and tail entity
        return: 
        pos1: a list of relative position of the head entity
        pos2: a list of relative position of the tail entitys
        """
        start1, end1, start2, end2 = pos['start1'],pos['end1'],pos['start2'],pos['end2']
        pos1_codes = []
        pos2_codes = []
        for idx, char in enumerate(sentence):
            # pos_code 是该字在句子中的相对位置
            pos1_code,pos2_code = self.get_pos_embeded(i = idx, lf1 = start1, rg1 = end1, lf2 = start2, rg2 = end2, maxlen =self.max_len)
            pos1_codes.append(pos1_code)
            pos2_codes.append(pos2_code)
        #pos_codes 是一个相对位置的列表，相当于是一个长度为len(sentence)的内容为每一字的相对位置的列表
        return pos1_codes,pos2_codes
    def symbolize_sentence(self, sentence):
        """
        input:
        sentence: a str, the sentence
        return:
        sentence: a list of char id"""
        mask = [1] * len(sentence)
        words = []
        length = min(self.max_len, len(sentence))
        mask = mask[:length]

        for i in range(length):
            words.append(self.char2id.get(sentence[i].lower(), self.char2id['UNK']))

        if length < self.max_len:
            for i in range(length, self.max_len):
                mask.append(0)  # 'PAD' mask is zero
                words.append(self.char2id['PAD'])

        unit = np.asarray([words, mask], dtype=np.int64)
        # unit = np.reshape(unit, newshape=(1, 2, self.max_len))
        return unit
class TextDataSet(BasicDataset):
    def __init__(self, rel2id,char2id, char_vec, config, filepath):
        """
        input:
        rel2id: a dict, key: relation name, value: relation id
        char2id: a dict, key: char, value: char id
        char_vec: a dict, key: char, value: char embedding
        config: a config class
        filepath: a str, the path of the dataset
        """
        super().__init__(rel2id,char2id,config)
        self.embed = nn.Embedding.from_pretrained(
            embeddings=char_vec,
            freeze= False
        )
        self.max_len = config.max_len
        self.data = []
        self.labels = []
        self.pos1 = []
        self.pos2 = []

        # load data
        with open(filepath, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.split('\t')
                
                if 'placeholder' in (line[0] or line[1]):
                    continue
                
                tmp = {}
                tmp['head'] = line[0]
                tmp['tail'] = line[1]
                tmp['relation'] = line[2]
                tmp['text'] = line[3][:-1]

                pos = {'start1': 0, 'end1': 0, 'start2': 0, 'end2': 0}
                #获取实体位置信息
                pos['start1'] = tmp['text'].find(tmp['head'],0)
                pos['end1'] = pos['start1'] + len(tmp['head'])
                pos['start2'] = tmp['text'].find(tmp['tail'],0)
                pos['end2'] = pos['start2'] + len(tmp['tail'])
                label = tmp['relation']
                sentence = tmp['text']
                
                label_idx = self.rel2id[label]
                one_sentence = super().symbolize_sentence(sentence)
                pos1, pos2 = super().add_position(sentence,pos)
                # pos是一个句子的相对位置列表
                self.pos1.append(np.array(super().position_padding(pos1)))
                self.pos2.append(np.array(super().position_padding(pos2)))
                self.data.append(one_sentence)
                self.labels.append(label_idx)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        pos1 = self.pos1[index]
        pos2 = self.pos2[index]

        return {'text':data,'label':label,'pos1':pos1,'pos2':pos2}

    def __len__(self):
        return len(self.data)


# 测试集是没有标签的，因此函数会略有不同
class TestDataSet(BasicDataset):
    def __init__(self, rel2id,char2id,char_vec,config,filepath):
        super().__init__(rel2id,char2id,config)
        self.embed = nn.Embedding.from_pretrained(
            embeddings=char_vec,
            freeze= False
        )
        self.max_len = config.max_len
        self.data = []
        self.pos1 = []
        self.pos2 = []
         # load data
        with open(filepath, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.split('\t')
                
                if 'placeholder' in (line[0] or line[1]):
                    continue
                
                tmp = {}
                tmp['head'] = line[0]
                tmp['tail'] = line[1]
                tmp['text'] = line[2][:-1]

                pos = {'start1': 0, 'end1': 0, 'start2': 0, 'end2': 0}
                #获取实体位置信息
                pos['start1'] = tmp['text'].find(tmp['head'],0)
                pos['end1'] = pos['start1'] + len(tmp['head'])
                pos['start2'] = tmp['text'].find(tmp['tail'],0)
                pos['end2'] = pos['start2'] + len(tmp['tail'])

                sentence = tmp['text']
                
                one_sentence = super().symbolize_sentence(sentence)
                pos1, pos2 = super().add_position(sentence,pos)
                # pos是一个句子的相对位置列表
                self.pos1.append(np.array(super().position_padding(pos1)))
                self.pos2.append(np.array(super().position_padding(pos2)))
                self.data.append(one_sentence)

    def __getitem__(self, index):
        data = self.data[index]
        pos1 = self.pos1[index]
        pos2 = self.pos2[index]

        return {'text':data,'pos1':pos1,'pos2':pos2}

    def __len__(self):
        return len(self.data)
class WordEmbeddingLoader(object):
    """
    A loader for pre-trained word embedding
    """

    def __init__(self, config):
        """
        input :
        config : config file
        """
        self.path_word = config.embedding_path  # path of pre-trained word embedding
        self.path_word = os.path.join(config.data_dir, self.path_word)
        self.word_dim = config.embedding_dimension  # dimension of word embedding

    def load_embedding(self):
        """
        load pre-trained word embedding
        output :
        word_vec : a dict, key is word and value is a numpy array of word embedding
        word2id : a dict, key is a word and value is a id
        """
        word2id = dict()  # word to wordID
        word_vec = list()  # wordID to word embedding

        word2id['PAD'] = len(word2id)  # PAD character
        word2id['UNK'] = len(word2id)  # out of vocabulary
        with open(self.path_word, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip().split()
                if len(line) != self.word_dim + 1:
                    continue
                word2id[line[0]] = len(word2id)
                word_vec.append(np.asarray(line[1:], dtype=np.float32))

        word_vec = np.stack(word_vec)
        vec_mean, vec_std = word_vec.mean(), word_vec.std()
        special_emb = np.random.normal(vec_mean, vec_std, (6, self.word_dim))
        special_emb[0] = 0  # <pad> is initialize as zero

        word_vec = np.concatenate((special_emb, word_vec), axis=0)
        word_vec = word_vec.astype(np.float32).reshape(-1, self.word_dim)
        word_vec = torch.from_numpy(word_vec)
        return word2id, word_vec
class RelationLoader(object):
    """
    A loader for relation
    """
    def __init__(self, config):
        self.data_dir = config.data_dir

    def __load_relation(self):
        """
        load relation
        output :
        rel2id : a dict, key is relation and value is a id
        id2rel : a dict, key is a id and value is a relation
        """
        relation_file = os.path.join(self.data_dir, 'rel2id.json')
        rel2id = {}
        id2rel = {}
        with open(relation_file, 'r', encoding='utf-8') as fr:
            file = json.load(fr)
            id2rel = file[0]
            rel2id = file[1]
        return rel2id, id2rel, len(rel2id)

    def get_relation(self):
        return self.__load_relation()
        
if __name__ == "__main__":
    config = Training_Config()
    char2id, char_vec = WordEmbeddingLoader(config).load_embedding()
    rel2id, id2rel, class_num = RelationLoader(config).get_relation()
    trainset = TextDataSet(rel2id,char2id,char_vec,config,filepath="./data/data_train.txt")
    testset = TestDataSet(rel2id,char2id,config,filepath="./data/test_exp3.txt")
    print("训练集长度为：", len(trainset))
    print("测试集长度为：", len(testset))
