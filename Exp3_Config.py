"""
该文件旨在配置训练过程中的各种参数
请按照自己的需求进行添加或者删除相应属性
"""


from tabnanny import check
from unittest import result
from torch import layer_norm


class Training_Config(object):
    def __init__(self,
                 embedding_dimension=100,
                 pos_embedding_dimension = 5,
                 vocab_size=20000,
                 training_epoch=10,
                 num_val=2,
                 max_sentence_length=40,
                 cuda=False,
                 label_num=2,
                 learning_rate=0.0001,
                 batch_size=10,
                 dropout=0.3,
                 embedding_path = "vec.txt",
                 data_dir = 'data',
                 hidden_size = 360,
                 layers_num = 2,
                 max_len = 300,
                 name = 'head_2_h_360',
                 checkpoints_dir = './checkpoints',
                 results_dir = './results',
                 gpu_ids = '1',
                 print_freq = 1000,
                 save_latest_freq = 500000,
                 save_by_iter = True,
                 epoch_decay = 20,
                 lr_policy = 'linear',
                 head_num = 2,
                 weight_decay = 1e-5):
        self.weight_decay = weight_decay
        self.head_num = head_num
        self.pos_embedding_dimension = pos_embedding_dimension
        self.lr_policy = lr_policy
        self.epoch_decay = epoch_decay
        self.save_by_iter = save_by_iter
        self.print_freq = print_freq
        self.save_latest_freq = save_latest_freq
        self.gpu_ids = gpu_ids
        self.checkpoints_dir = checkpoints_dir
        self.results_dir = results_dir
        self.name = name
        self.max_len = max_len
        self.layers_num = layers_num
        self.hidden_size = hidden_size # the dimension of hidden units in (Bi)LSTM layer
        self.embedding_path = embedding_path
        self.data_dir = data_dir
        self.embedding_dimension = embedding_dimension  # 词向量的维度
        self.vocab_size = vocab_size  # 词汇表大小
        self.epoch = training_epoch  # 训练轮数
        self.num_val = num_val  # 经过几轮才开始验证
        self.max_sentence_length = max_sentence_length  # 句子最大长度
        self.label_num = label_num  # 分类标签个数
        self.lr = learning_rate  # 学习率
        self.batch_size = batch_size  # 批大小
        self.cuda = cuda  # 是否用CUDA
        self.dropout = dropout  # dropout概率

