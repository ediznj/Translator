self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
self.src_posi_embedding = nn.Embedding(max_len, embedding_size)
src_positions = (
    torch.arange(0, src_len).unsqueeze(-1).expand(src.shape)
)
embed_src = self.dropout(
    self.src_word_embedding(src) + self.src_posi_embedding(src_positions)
)

import torch.nn as nn
import torch

def get_embed(word_embed, vocab_size, embed_size):
    """
    :param src: shape=> batch x seq_len
    :return: batch x seq_len x embed_size
    """
    assert word_embed.dim() == 2, f'Unsupported input dimension size'
    src = word_embed
    batch = src.size(0)
    seq_len = src.size(1)
    w_embed = nn.Embedding(vocab_size, embed_size)
    p_embed = nn.Embedding(seq_len, embed_size)
    src_we = w_embed(src)
    src_pe = p_embed(torch.arange(seql_len).unsqueeze(-1).expand(src.shape))
    src = nn.Dropout(src_we + src_pe)
    return src, src.shape



src_pad_mask = self.make_src_mask(src)
tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_len).to(device)