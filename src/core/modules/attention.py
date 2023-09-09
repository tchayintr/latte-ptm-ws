import torch
import torch.nn as nn
from allennlp_light.nn.util import masked_softmax


class Attention(nn.Module):
    def __init__(self,
                 embed_size: int,
                 hidden_size: int,
                 attn_comp_type: str = 'wavg',
                 inner_dropout: float = 0.0):
        super(Attention, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.attn_comp_type = attn_comp_type
        self.inner_dropout = inner_dropout

        self.W = nn.Linear(embed_size, hidden_size)
        self.dropout = nn.Dropout(inner_dropout)

        if attn_comp_type != 'wavg' and attn_comp_type != 'wcon':
            raise ValueError('invalid argument for --attn_comp_type')

    def forward(self, char_hs, word_es, word_mask):
        '''
        char_hs: (batch_size, num_chars, hidden_size)
        word_hs: (batch_size, num_chars, num_candidates, word_embed_size)
        word_mask: (batch_size, num_chars, num_candidates)
          -> True: word, False: <pad>

        chars
        c1  c2  c3  c4  c5
          -> chars.shape == (B,N=5,H)

        word candidates (k=3)
        char | candidate words (num_candidates: L = \Sum_1^k k)
               k=1  k=2          k=3
        c1   | c1   <pad> c1c2   <pad>  <pad>  c1c2c3
        c2   | c2   c1c2  c2c3   <pad>  c1c2c3 c2c3c4
        c3   | c3   c2c3  c3c4   c1c2c3 c2c3c4 c3c4c5
        c4   | c4   c3c4  c4c5   c2c3c4 c3c4c5 <pad>
        c5   | c5   c4c5  <pad>  c3c4c5 <pad>  <pad>
          -> word_hs.shape == (B, N=5, L=6(1+2+3), E)

        word_scores (S=sum)
        S(c1*c1) S(c1*<pad>) S(c1*c1c2 ) S(c1*<pad> ) S(c1*<pad> ) S(c1*c1c2c3)
        S(c2*c2) S(c2*c1c2 ) S(c2*c2c3 ) S(c2*<pad> ) S(c2*c1c2c3) S(c2*c2c3c4)
        S(c3*c3) S(c3*c2c3 ) S(c3*c3c4 ) S(c3*c1c2c3) S(c3*c2c3c4) S(c3*c3c4c5)
        S(c4*c4) S(c4*c3c4 ) S(c4*c4c5 ) S(c4*c2c3c4) S(c4*c3c4c5) S(c4*<pad> )
        S(c5*c5) S(c5*c4c5 ) S(c5*<pad>) S(c5*c3c4c5) S(c5*<pad> ) S(c5*<pad> )
        ;where, c1, c1c2, ..., and c1*c1c2: vector, S(c1*c1c2):
            scalar (similairty between c1 and c1c2)
        '''
        '''
        transfrom every candidate token features into the same dimension
        as a character dimension
        '''
        word_hs = self.W(word_es)
        '''compute candidate token scores for each character'''
        word_scores = torch.einsum('bnh,bnlh->bnl', char_hs, word_hs)
        '''
        compute weight of candidate tokens for each candidate token
        of each character by using softmax function
        without considering masked slot in word_mask
        '''
        word_weight = masked_softmax(word_scores, word_mask, dim=2)
        '''
        apply inner dropout to candidate token features and
        compute summary matrix (weight . candidate token features)
        '''
        weighted_word_hs = torch.einsum('bnlh,bnl->bnlh',
                                        self.dropout(word_hs), word_weight)

        if self.attn_comp_type == 'wcon':
            '''v.shape == (B, N, L * H)'''
            B, N, L, H = word_hs.shape
            v = weighted_word_hs.view(B, N, L * H)
        elif self.attn_comp_type == 'wavg':
            '''v.shape == (B, N, H)'''
            v = weighted_word_hs.sum(2)
        '''concat character features with summary matrix'''
        return torch.cat([char_hs, v], dim=2), word_weight

    def _forward_check(self, char_batch, word_batch, word_mask):
        v = []
        for b_idx, (char_hs, word_es_for_chars,
                    mask) in enumerate(zip(char_batch, word_batch, word_mask)):
            word_hs_for_chars = self.W(word_es_for_chars)
            # char_hs: (num_chars, hidden_size)
            # word_hs: (num_chars, num_words, hidden_size)
            # mask: (num_chars, num_words)

            scores = []
            for char_h, word_hs in zip(char_hs, word_hs_for_chars):
                # char_h: (hidden_size)
                # word_hs: (num_words, hidden_size)
                _scores = []
                for word_h in word_hs:
                    score = sum(char_h * word_h)
                    _scores.append(score)

                _scores = torch.tensor(_scores, dtype=torch.float)
                # _scores: (num_words)
                scores.append(_scores)

            scores = torch.stack(scores)
            # scores: (num_chars, num_words)
            weight = masked_softmax(scores, mask, dim=1)
            # weight = self.dropout(weight)
            # weight: (num_chars, num_words)

            weighted_word_hs_for_chars = []
            for w, word_hs in zip(weight, word_hs_for_chars):
                # w: num_words
                # word_hs: (num_words, hidden_size)
                w = w.unsqueeze(-1)
                weighted_word_hs = w * self.dropout(word_hs)
                weighted_word_hs_for_chars.append(weighted_word_hs)

            weighted_word_hs_for_chars = torch.stack(
                weighted_word_hs_for_chars)
            # weighted_word_hs_for_chars: (num_words, hidden_size)

            _v = None
            if self.attn_comp_type == 'wcon':
                num_chars, num_words, hidden_size = word_hs_for_chars.shape
                _v = weighted_word_hs_for_chars.view(num_chars,
                                                     num_words * hidden_size)
                # _v: (num_chars, num_words*hidden_size)
            elif self.attn_comp_type == 'wavg':
                _v = weighted_word_hs_for_chars.sum(1)
                # _v: (num_chars, hidden_size)

            v.append(_v)

        v = torch.stack(v)
        # v: (num_chars, hidden_size) or (num_chars, num_words*hidden_size)

        return torch.cat([char_batch, v], dim=2)


if __name__ == '__main__':
    # word_embed_size = 300
    # hidden_size = 256
    word_embed_size = 5  # Hw
    hidden_size = 2  # H
    attn_comp_type = 'wavg'
    dropout = 0
    att = Attention(word_embed_size, hidden_size, attn_comp_type, dropout)
    print(att)

    # input
    # BATCH_SIZE = 5
    # NUM_CHARS = 10
    # k = 3
    # NUM_CANDIDATES = 3 + 2 + 1
    BATCH_SIZE = 1  # B
    NUM_CHARS = 4  # N
    k = 3
    NUM_CANDIDATES = 3 + 2 + 1  # L
    char_hs = torch.rand(BATCH_SIZE, NUM_CHARS, hidden_size)
    word_es = torch.rand(BATCH_SIZE, NUM_CHARS, NUM_CANDIDATES,
                         word_embed_size)
    word_indices = torch.randint(0, 100,
                                 (BATCH_SIZE, NUM_CHARS, NUM_CANDIDATES))
    # word_mask = word_indices < 3
    word_mask = word_indices < 50

    # equality check
    v, w = att(char_hs, word_es, word_mask)
    _v = att._forward_check(char_hs, word_es, word_mask)
    print('v:', v.shape, '_v:', _v.shape)
    print(torch.allclose(v, _v))
