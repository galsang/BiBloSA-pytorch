import torch
import torch.nn as nn
import torch.nn.functional as F


class BiBloSA(nn.Module):
    def __init__(self, args, data):
        super(BiBloSA, self).__init__()

        self.args = args

        self.word_emb = nn.Embedding(args.word_vocab_size, args.word_dim)
        # initialize word embedding with GloVe
        self.word_emb.weight.data.copy_(data.TEXT.vocab.vectors)
        # fine-tune the word embedding
        self.word_emb.weight.requires_grad = True

    def reset_parameters(self):
        # <unk> vectors is randomly initialized
        nn.init.uniform(self.word_emb.weight.data[0], -0.05, 0.05)

    def dropout(self, v):
        return F.dropout(v, p=self.args.dropout, training=self.training)

    def forward(self, batch):
        p = self.word_emb(batch.premise)
        h = self.word_emb(batch.hypothesis)