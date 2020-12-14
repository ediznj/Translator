import torch
import torch.nn as nn
from torch.autograd import Variable as V

expected = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 0], [0, 5, 6, 7, 8, 1, 2, 3, 4, 0]])
expected = expected[:, 1:]  # .reshape(-1)  # , batch)
# assert 1==0, f'{expected.shape} {expected.size()}'
"""
Parameters Found:
batch_size, embed_size,
src_max_len, tgt_max_len, src_vocab_size, tgt_vocab_size
embed_size, heads, num_encoders, num_decoders, ffn_size, dropout

src_embed = nn.Embedding(src_vocab_size, embed_size)
tgt_embed = nn.Embedding(tgt_vocab_size, embed_size)
transformer = nn.Transformers(embed_size, heads, num_encoders, num_decoders, ffn_size, dropout)

src.shape = (batch_size, src_max_len, src_vocab_size)
tgt.shape = (batch_size, tgt_max_len, src_vocab_size)
src = 
tgt = 

"""


def _translator(src=None, tgt=None, embed_size=512):
    """
    src: expected tensor shape (src_maxseq_len, batch_size, src_vocab_size)
    tgt: expected tensor shape (tgt_maxseq_len, batch_size, tgt_vocab_size)
    :rtype: torch.tensor of shape=(tgt_maxseq_len, batch_size, embed_size)
    """
    # nn.Transformer expects structure of src & tgt: seq_len x batch x embed_size
    if not (src and tgt):
        batch, embed_size = 2, 12  # also embed_size == dmodel
        src_maxseq_len = 8
        src_vocab_size = 20
        src = torch.randint(20, (src_maxseq_len, batch, src_vocab_size))  # .type(torch.float)
        tgt_maxseq_len = 10
        tgt_vocab_size = 30
        tgt = torch.randint(30, (tgt_maxseq_len, batch, tgt_vocab_size))  # .type(torch.float)
        tgt[0, :, :], tgt[-1, :, :] = 0, 0
    else:
        src_maxseq_len, batch, src_vocab_size = src.shape()
        tgt_maxseq_len, batch, tgt_vocab_size = tgt.shape()
        # src, tgt = input_adaptor_transformer(src, tgt)

    # src.requires_grad=False #V(src)
    # tgt.requires_grad=False #V(tgt)
    src_embed = nn.Embedding(src_vocab_size, embed_size)
    tgt_embed = nn.Embedding(tgt_vocab_size, embed_size)

    print(f'src:{src.shape} \t tgt:{tgt.shape}')
    src = src_embed(src.argmax(-1))
    tgt = tgt_embed(tgt.argmax(-1))
    src, tgt = V(src), V(tgt)
    # dmodel = 12, heads=3, encoders=1, decoders=1, ffn=12, dropout=0
    model = nn.Transformer(embed_size, 3, 1, 1, 12, 0)
    tgt_mask = model.generate_square_subsequent_mask(tgt_maxseq_len - 1)

    learning_rate = 0.01
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # for k,v in model.named_parameters():
    #   print(f'key:{k}')
    #   print(f'value:{v}')

    num_epoch = 50
    model.train()
    try:
        for epoch in range(num_epoch):
            print(f'Epoch #{epoch + 1}/{num_epoch}')
            assert (src.dim() == 3 and tgt.dim() == 3), f'Error unsupported input dimensions'
            assert (src.size(1) == tgt[:-1].size(1)), f'Batch Size Error src{src.shape} and tgt{tgt.shape}'
            assert (src.size(2) == embed_size and tgt[:-1].size(2) == embed_size) \
                , f'Error src{src.shape} and tgt{tgt.shape}'
            out = model(src, tgt[:-1], tgt_mask=tgt_mask)
            out = out.reshape(batch, -1, embed_size)
            info = f'out.size:{out.size()} expcted.size:{expected.size()}'
            # assert (out.size(0) == expected.size(0)), info
            print(info)

            # seq_len, batch, tgt_vocab_size = out.shape(0), out.shape(1), out.shape(2)
            print(f'Calculating Loss')
            loss = criterion(out.reshape(-1, embed_size), expected.reshape(-1))

            print(f'Loss: {loss}')
            optim.zero_grad()
            loss.backward()
            optim.step()
        return out
    except ValueError:
        # print('caught exception: ', ValueError)
        pass


def translator_rolled(src=None, tgt=None):
    # nn.Transformer expects structure of src & tgt: seq_len x batch x embed_size
    batch, embed_size = 1, 12  # also embed_size == dmodel
    src_maxseq_len, tgt_maxseq_len = 8, 10

    # V = lambda x: Variable(x)
    if not src:
        src = torch.randint(embed_size, (src_maxseq_len, batch, embed_size)).type(torch.float)
    if not tgt:
        tgt = torch.randint(embed_size, (tgt_maxseq_len, batch, embed_size)).type(torch.float)
    tgt[0, :, :], tgt[-1, :, :] = 0, 0

    src_var = V(src)
    tgt_var = V(tgt)

    # dmodel = 12, heads=3, encoders=1, decoders=1, ffn=12, dropout=0
    model = nn.Transformer(embed_size, 3, 1, 1, 12, 0)
    tgt_mask = model.generate_square_subsequent_mask(tgt_maxseq_len - 1)

    learning_rate = 0.01
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    num_epoch = 20
    model.train()
    for epoch in range(num_epoch):
        print(f'Epoch #{epoch + 1}/{num_epoch}')
        assert (src.size(1) == tgt[:-1].size(1))
        assert (src.size(2) == embed_size and tgt[:-1].size(2) == embed_size)
        out = model(src_var, tgt_var[:-1], tgt_mask=tgt_mask)
        assert (out.size(0) == expected.size(0))
        out = out.reshape(-1, embed_size)
        loss = criterion(out, expected)
        print(f'Loss: {loss}')
        optim.zero_grad()
        loss.backward()
        optim.step()
    return out


def unrolled_translator():
    # nn.Transformer expects structure of src & tgt: seq_len x batch x embed_size
    batch, embed_size = 1, 12  # also embed_size == dmodel
    src_maxseq_len, tgt_maxseq_len = 8, 10

    # V = lambda x: Variable(x)

    src = torch.randint(embed_size, (src_maxseq_len, batch, embed_size)).type(torch.float)
    tgt = torch.randint(embed_size, (tgt_maxseq_len, batch, embed_size)).type(torch.float)
    tgt[0, :, :], tgt[-1, :, :] = 0, 0
    expected = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 0]])  # , [0, 5,6,7,8,1,2,3,4, 0]])
    expected = expected[:, 1:].reshape(-1)  # , batch)

    src_var = V(src)
    tgt_var = V(tgt)

    dmodel = 12  # , heads=3, encoders=1, decoders=1, ffn=12, dropout=0
    model = nn.Transformer(dmodel, 3, 1, 1, 12, 0)
    tgt_mask = model.generate_square_subsequent_mask(tgt_maxseq_len - 1)

    learning_rate = 0.01
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    print('EPOCH NO 1 -> Arbitrary Output')
    assert (src.size(1) == tgt[:-1].size(1))
    assert (src.size(2) == dmodel and tgt[:-1].size(2) == dmodel)
    out = model(src_var, tgt_var[:-1], tgt_mask=tgt_mask)
    assert (out.size(0) == expected.size(0))
    out = out.reshape(-1, embed_size)
    loss = criterion(out, expected)
    print(f'Loss: {loss}')
    optim.zero_grad()
    loss.backward()
    optim.step()

    print('EPOCH NO 2 -> Post Gradient/Weight Adjustment')
    assert (src.size(1) == tgt[:-1].size(1))
    assert (src.size(2) == dmodel and tgt[:-1].size(2) == dmodel)
    out = model(src_var, tgt_var[:-1], tgt_mask=tgt_mask)
    assert (out.size(0) == expected.size(0))
    out = out.reshape(-1, embed_size)
    loss = criterion(out, expected)
    print(f'Loss: {loss}')
    optim.zero_grad()
    loss.backward()
    optim.step()

    print('EPOCH NO 3 -> Post Gradient/Weight Adjustment')
    assert (src.size(1) == tgt[:-1].size(1))
    assert (src.size(2) == dmodel and tgt[:-1].size(2) == dmodel)
    out = model(src_var, tgt_var[:-1], tgt_mask=tgt_mask)
    assert (out.size(0) == expected.size(0))
    out = out.reshape(-1, embed_size)
    loss = criterion(out, expected)
    print(f'Loss: {loss}')
    optim.zero_grad()
    loss.backward()
    optim.step()


class Data(object):
    class TextData(object):
        def __init__(self, name, data):
            self.name = name
            self.data = data

        def __str__(self):
            print('\n__str__ called')
            return f'\nName:{self.name} \n Data: {self.data}'

        def __repr__(self):
            print('\n__repr__ called')
            return f'Name: {self.name}, Data: {self.data}'

    def __init__(self):
        print('init Data')
        self.src = None
        self.tgt = None

    def source(self, name, text):
        print('inside src')
        self.src = Data.TextData(name, text)
        return self

    def target(self, name, text):
        print('inside tgt')
        self.tgt = Data.TextData(name, text)
        return self

    def get(self):
        print(f'inside Data.get')
        print(self.src, self.tgt)


def get_embed(word_embed, vocab_size, embed_size):
    """
    :param src: shape=> seq_len x batch
    :return:  seq_len x batch x embed_size
    """
    assert word_embed.dim() == 2, f'Unsupported input dimension size'
    src = word_embed
    seq_len = src.size(0)
    batch = src.size(1)
    w_embed = nn.Embedding(vocab_size, embed_size)
    p_embed = nn.Embedding(seq_len, embed_size)
    src_we = w_embed(src)
    src_pe = p_embed(torch.arange(seq_len).unsqueeze(1).expand(seq_len, batch))
    # assert 1 == 0, f'src: {src_we.shape} tgt: {src_pe.shape}'
    dropout = nn.Dropout(0.10)
    src = dropout(src_we + src_pe)
    return src, src.shape

if __name__ == '__main__':
    # unrolled_translator()
    # out = translator_rolled()
    # out = _translator()
    # print(f'out: {out.argmax(-1)}')
    # print(f'Expected: {expected}')
    # t = Data().source('en', 'English').target('ta', 'Tamil')
    # t.get()
    vocab_size = 30; max_sent_len=9; batch = 5; embed_size = 15
    text_embed = torch.randint(vocab_size, (max_sent_len, batch))
    print(text_embed.shape)
    # import pdb; pdb.set_trace()
    # out = torch.arange(max_sent_len).unsqueeze(-1).expand(text_embed.shape)
    # print(out)
    out, sz = get_embed(text_embed, vocab_size, embed_size)
    print(out, sz)
    print(out.argmax(-1))
