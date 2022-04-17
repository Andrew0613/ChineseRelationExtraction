from numpy import float128, float64
import torch
import torch.nn as nn
import torch.nn.functional as functions
from Exp3_Config import Training_Config
from Exp3_DataSet import WordEmbeddingLoader, RelationLoader
from torch.nn import init
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
def initialize_weight(x):
    """
    initialize the weight of the model
    """
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)
def create_pad_mask(t, pad):
    mask = (t == pad).unsqueeze(-2)
    return mask
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, dropout_rate, head_size=8):
        """
        input:
            hidden_size: the size of the hidden state
            dropout_rate: the dropout rate
            head_size: the number of heads
        """
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size, bias=False)
        initialize_weight(self.linear_q)
        initialize_weight(self.linear_k)
        initialize_weight(self.linear_v)

        self.att_dropout = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size,
                                      bias=False)
        initialize_weight(self.output_layer)

    def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask:torch.Tensor, cache=None):
        """
        input:
            q: the query, shape: [batch_size, seq_len, hidden_size]
            k: the key, shape: [batch_size, seq_len, hidden_size]
            v: the value, shape: [batch_size, seq_len, hidden_size]
            mask: the mask of the padding, shape: [batch_size, seq_len]
            cache: the cache of the previous attention, shape: [batch_size, seq_len, head_size, att_size]
        output:
            attention, shape: [batch_size, seq_len, hidden_size]"""
        orig_q_size = q.size()
        orig_q = q
        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        if cache is not None and 'encdec_k' in cache:
            k, v = cache['encdec_k'], cache['encdec_v']
        else:
            k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
            v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

            if cache is not None:
                cache['encdec_k'], cache['encdec_v'] = k, v
        
        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]
        
        
        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q.mul_(self.scale)
        
        
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        # mask = mask.unsqueeze(-1).float()
        # score_mask = torch.matmul(mask,mask.transpose(1,2))
        x.masked_fill_(mask.unsqueeze(1), -1e9)
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x
class TextCNN_Model(nn.Module):

    def __init__(self, configs, char_vec,char_id, class_num):
        """
        input:
            configs: the configuration of the model
            char_vec: the char embedding of the model
            char_id: the char id of the model
            class_num: the number of the class
        """
        super(TextCNN_Model, self).__init__()
        self.gpu_ids = configs.gpu_ids
        self.model_name = "BiLSTM_Model"
        self.hidden_size = configs.hidden_size
        vocab_size = configs.vocab_size
        self.layers_num = configs.layers_num
        embedding_dimension = configs.embedding_dimension
        pos_embedding_dimension = configs.pos_embedding_dimension
        label_num = configs.label_num
        self.class_num = class_num
        self.max_len = configs.max_len
        pos_size = 2*configs.max_len + 3
        self.char_vec = char_vec
        self.char_id = char_id
        # 词嵌入和dropout
        # self.embed = nn.Embedding(vocab_size, embedding_dimension)
        self.embed = nn.Embedding.from_pretrained(
            embeddings=char_vec,
            freeze= False
        )
        self.pos1_embeds = nn.Embedding(pos_size,pos_embedding_dimension)
        self.pos2_embeds = nn.Embedding(pos_size,pos_embedding_dimension)
        self.dropout = nn.Dropout(configs.dropout)
        self.emb_dropout = nn.Dropout(configs.dropout)
        self.lstm_dropout = nn.Dropout(configs.dropout)
        self.linear_dropout = nn.Dropout(configs.dropout)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(
            input_size=embedding_dimension + 2*pos_embedding_dimension,
            hidden_size=self.hidden_size,
            num_layers=self.layers_num,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=True,
        )
        self.multi_attention = MultiHeadAttention(hidden_size=configs.hidden_size,dropout_rate=configs.dropout,head_size = configs.head_num)
        self.att_weight = nn.Parameter(torch.randn(1, self.hidden_size, 1))
        
        self.dense = nn.Linear(
        in_features=self.hidden_size,
        out_features=self.class_num,
        bias=True
    )
        self.multi_dense = nn.Linear(
        in_features=self.hidden_size*self.max_len,
        out_features=self.class_num,
        bias=True
    )
        # initialize weight
        init.xavier_normal_(self.dense.weight)
        init.constant_(self.dense.bias, 0.)

        init.xavier_normal_(self.multi_dense.weight)
        init.constant_(self.multi_dense.bias, 0.)

    def lstm_layer(self, x, mask):
        """
        input:
            x: [batch_size, seq_len, embedding_dimension]
            mask: [batch_size, seq_len]
        output:
            x: [batch_size, seq_len, hidden_size]
        """
        tmp = mask.gt(0)
        lengths = torch.sum(mask.gt(0), dim=-1)
        x_packed = pack_padded_sequence(x, lengths.to("cpu"), batch_first=True, enforce_sorted=False)#pad只是为了等长输入进模型，但是pad进模型会产生误差以及浪费算力，所以会将原tensor进行压缩
        # print(x_packed)
        h, (_, _) = self.lstm(x_packed)
        h, _ = pad_packed_sequence(h, batch_first=True, padding_value=0.0, total_length=self.max_len)
       
        h = h.view(-1, self.max_len, 2, self.hidden_size)
        h = torch.sum(h, dim=2)  # B*L*H
        return h

    # def attention_layer(self, h, mask):
    #     att_weight = self.att_weight.expand(mask.shape[0], -1, -1)  # B*H*1
    #     att_score = torch.bmm(self.tanh(h), att_weight)  # B*L*H  *  B*H*1 -> B*L*1

    #     # mask, remove the effect of 'PAD'
    #     mask = mask.unsqueeze(dim=-1)  # B*L*1
    #     # tmp = mask.eq(0)
    #     # tmp1 =mask.gt(0)
    #     att_score = att_score.masked_fill(mask.eq(0), float('-inf'))  # B*L*1
    #     att_weight = F.softmax(att_score, dim=1)  # B*L*1

    #     reps = torch.bmm(h.transpose(1, 2), att_weight).squeeze(dim=-1)  # B*H*L *  B*L*1 -> B*H*1 -> B*H
    #     reps = self.tanh(reps)  # B*H
    #     return reps

    def forward(self, data,pos1,pos2):
        token = data[:, 0, :].view(-1, self.max_len)
        c_mask = create_pad_mask(token,pad = self.char_id['PAD'])
        mask = data[:, 1, :].view(-1, self.max_len)
        pos1 = self.pos1_embeds(pos1.long())
        pos2 = self.pos2_embeds(pos2.long())
        emb = self.embed(token)  # B*L*word_dim
        
        emb = torch.cat((emb,pos1,pos2),2) # [B, L, word_dim+2*pos_dim]
        emb = self.emb_dropout(emb)
        h = self.lstm_layer(emb, mask)  # B*L*H
        h = self.lstm_dropout(h)
        attention_output = self.multi_attention(h,h,h,c_mask) # B*L*H
        # reps = self.attention_layer(h, mask)  # B*H
        # reps = h.permute(0,2,1)
        # reps = self.linear_dropout(attention_output)
        attention_output = attention_output.view(-1,self.max_len*self.hidden_size)
        logits = self.multi_dense(attention_output)
        # logits = self.dense(reps) 
        return logits

if __name__ == "__main__":
    config = Training_Config()
    char2id, char_vec = WordEmbeddingLoader(config).load_embedding()
    rel2id, id2rel, class_num = RelationLoader(config).get_relation()
    model = TextCNN_Model(configs=config,char_vec=char_vec,class_num = class_num)
