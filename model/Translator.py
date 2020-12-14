import torch
import torch.nn as nn
from torch.autograd import Variable

class Data(object):
    def __init__(self):
        pass

class TextData(object):
    def __init__(self, name, data):
        # super().__init__()
        self.name = name
        self.data = data

    def __str__(self):
        print('\n__str__ called')
        return f'\nName:{self.name} \n Data: {self.data}'

    def __repr__(self):
        print('\n__repr__ called')
        return f'Name: {self.name}, Data: {self.data}'


def run_train(data):
    pass


def run_eval(data):
    pass


def predict():
    pass


class Translator(nn.Module):
    """USAGE: TBD"""

    def __init__(self,
                 src_vocab_size,
                 tgt_vocab_size,
                 embed_size,
                 heads=3,
                 num_encoders=3,
                 num_decoders=3,
                 ffn_out=2048,
                 dropout=0.10
                 ):
        super().__init__()
        self._name = 'Translator using torch.nn.Transformer'
        self.src_embed = nn.Embedding(src_vocab_size, embed_size)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, embed_size)
        self.transformer = nn.Transformer(embed_size,
                                          heads,
                                          num_encoders,
                                          num_decoders,
                                          ffn_out,
                                          dropout
                                          )
        self.fc_out = nn.Linear(embed_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    # def source(self):
    #     pass
    # def target(self):
    #     pass

    # @staticmethod
    # def get_we(src, tgt):
    #     assert src.dim() > 3, 'src dimension greater than expected'
    #     assert tgt.dim() > 3, 'src dimension greater than expected'
    #     if src.dim() == 3 and tgt.dim() == 3:
    #         # assuming dim0->batch_size,  dim1->seq_len,  dim2->vocab_size
    #         src_we, tgt_we = src.argmax(-1), tgt.argmax(-1)
    #         src_pe, tgt_pe = None, None
    #
    # @staticmethod
    # def get_pe(src, tgt):
    #     batch_size, seq_len = src.

    def forward(self, src, tgt):
        src_we, tgt_we = Translator.get_we(src,tgt)
        src_pe, tgt_pe =
        self.src_embed()
        self.tgt_embed()
        pass

# class Translator_Factory:
#     def __init__(self):
#         self.translator = None
#         self._train_data = {}
#
#     @staticmethod
#     def get_translator(self, name=None):
#         if not name:
#             return Translator()
#
#     @property
#     def train_data(self):
#         return self._train_data
#
#     @train_data.setter
#     def train_data(self, train_data):
#         # add train_data validation here
#         self._train_data = train_data
#
#     def start_train(self, train_data):
#         self.train_data = train_data
#         run_train(self.train_data)
