import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class customizedModule(nn.Module):
    def __init__(self):
        super(customizedModule, self).__init__()

    # linear transformation (w/ initialization) + activation + dropout
    def customizedLinear(self, in_dim, out_dim, activation=None, dropout=False):
        cl = nn.Sequential(nn.Linear(in_dim, out_dim))
        nn.init.xavier_uniform(cl[0].weight)
        nn.init.constant(cl[0].bias, 0)

        if activation is not None:
            cl.add_module(str(len(cl)), activation)
        if dropout:
            cl.add_module(str(len(cl)), nn.Dropout(p=self.args.dropout))

        return cl


class NN4SNLI(customizedModule):
    def __init__(self, args, data):
        super(NN4SNLI, self).__init__()

        self.args = args
        # set hyperparameters
        # r: length of inner blocks
        self.args.r = self.args.block_size
        self.args.c = self.args.mSA_scalar

        self.word_emb = nn.Embedding(args.word_vocab_size, args.word_dim)
        # initialize word embedding with GloVe
        self.word_emb.weight.data.copy_(data.TEXT.vocab.vectors)
        # fine-tune the word embedding
        self.word_emb.weight.requires_grad = True
        # <unk> vectors is randomly initialized
        nn.init.uniform(self.word_emb.weight.data[0], -0.05, 0.05)

        self.BiBloSAN = BiBloSAN(self.args)
        self.fc = self.customizedLinear(self.args.word_dim * 2 * 4, 300, activation=nn.ReLU(), dropout=True)
        self.fc_softmax = self.customizedLinear(300, self.args.class_size)

    def forward(self, batch):
        # (batch, seq_len, word_dim)
        p = self.word_emb(batch.premise)
        h = self.word_emb(batch.hypothesis)

        # (batch, word_dim * 2)
        p = self.BiBloSAN(p)
        h = self.BiBloSAN(h)

        x = F.dropout(torch.cat([p, h, p * h, torch.abs(p - h)], dim=1), p=self.args.dropout, training=self.training)
        x = self.fc(x)
        x = self.fc_softmax(x)

        return x


class BiBloSAN(customizedModule):
    def __init__(self, args):
        super(BiBloSAN, self).__init__()

        self.args = args

        self.mBloSA_fw = mBloSA(self.args, 'fw')
        self.mBloSA_bw = mBloSA(self.args, 'bw')

        # two untied fully connected layers
        self.fc_fw = self.customizedLinear(self.args.word_dim, self.args.word_dim, activation=nn.ReLU())
        self.fc_bw = self.customizedLinear(self.args.word_dim, self.args.word_dim, activation=nn.ReLU())

        self.s2tSA = s2tSA(self.args, self.args.word_dim * 2)

    def forward(self, x):
        input_fw = self.fc_fw(x)
        input_bw = self.fc_bw(x)

        # (batch, seq_len, word_dim)
        u_fw = self.mBloSA_fw(input_fw)
        u_bw = self.mBloSA_bw(input_bw)

        # (batch, seq_len, word_dim * 2) -> (batch, word_dim * 2)
        u_bi = self.s2tSA(torch.cat([u_fw, u_bw], dim=2))
        return u_bi


class mBloSA(customizedModule):
    def __init__(self, args, mask):
        super(mBloSA, self).__init__()

        self.args = args
        self.mask = mask

        # init submodules
        self.s2tSA = s2tSA(self.args, self.args.word_dim)
        self.init_mSA()
        self.init_mBloSA()

    def init_mSA(self):
        self.m_W1 = self.customizedLinear(self.args.word_dim, self.args.word_dim)
        self.m_W2 = self.customizedLinear(self.args.word_dim, self.args.word_dim)
        self.m_b = nn.Parameter(torch.zeros(self.args.word_dim))

        self.m_W1[0].bias.requires_grad = False
        self.m_W2[0].bias.requires_grad = False

        self.c = nn.Parameter(torch.Tensor([self.args.c]), requires_grad=False)

    def init_mBloSA(self):
        self.g_W1 = self.customizedLinear(self.args.word_dim, self.args.word_dim)
        self.g_W2 = self.customizedLinear(self.args.word_dim, self.args.word_dim)
        self.g_b = nn.Parameter(torch.zeros(self.args.word_dim))

        self.g_W1[0].bias.requires_grad = False
        self.g_W2[0].bias.requires_grad = False

        self.f_W1 = self.customizedLinear(self.args.word_dim * 3, self.args.word_dim, activation=nn.ReLU())
        self.f_W2 = self.customizedLinear(self.args.word_dim * 3, self.args.word_dim)

    def mSA(self, x):
        """
        masked self-attention module
        :param x: (batch, (block_num), seq_len, word_dim)
        :return: s: (batch, (block_num), seq_len, word_dim)
        """
        seq_len = x.size(-2)

        # (batch, (block_num), seq_len, 1, word_dim)
        x_i = self.m_W1(x).unsqueeze(-2)
        # (batch, (block_num), 1, seq_len, word_dim)
        x_j = self.m_W2(x).unsqueeze(-3)

        # build fw or bw masking
        # (seq_len, seq_len)
        M = Variable(torch.ones((seq_len, seq_len))).cuda(self.args.gpu).triu().detach()
        M[M == 1] = float('-inf')

        # CASE 1 - x: (batch, seq_len, word_dim)
        # (1, seq_len, seq_len, 1)
        M = M.contiguous().view(1, M.size(0), M.size(1), 1)
        # (batch, 1, seq_len, word_dim)
        # padding to deal with nan
        pad = torch.zeros(x.size(0), 1, x.size(-2), x.size(-1))
        pad = Variable(pad).cuda(self.args.gpu).detach()

        # CASE 2 - x: (batch, block_num, seq_len, word_dim)
        if len(x.size()) == 4:
            M = M.unsqueeze(1)
            pad = torch.stack([pad] * x.size(1), dim=1)

        # (batch, (block_num), seq_len, seq_len, word_dim)
        f = self.c * F.tanh((x_i + x_j + self.m_b) / self.c)

        # fw or bw masking
        if f.size(-2) > 1:
            if self.mask == 'fw':
                M = M.transpose(-2, -3)
                f = F.softmax((f + M).narrow(-3, 0, f.size(-3) - 1), dim=-2)
                f = torch.cat([f, pad], dim=-3)
            elif self.mask == 'bw':
                f = F.softmax((f + M).narrow(-3, 1, f.size(-3) - 1), dim=-2)
                f = torch.cat([pad, f], dim=-3)
            else:
                raise NotImplementedError('only fw or bw mask is allowed!')
        else:
            f = pad

        # (batch, (block_num), seq_len, word_dim)
        s = torch.sum(f * x.unsqueeze(-2), dim=-2)
        return s

    def forward(self, x):
        """
        masked block self-attention module
        :param x: (batch, seq_len, word_dim)
        :param M: (seq_len, seq_len)
        :return: (batch, seq_len, word_dim)
        """
        r = self.args.r
        n = x.size(1)
        m = n // r

        # padding for the same length of each block
        pad_len = (r - n % r) % r
        if pad_len:
            pad = Variable(torch.zeros(x.size(0), pad_len, x.size(2))).cuda(self.args.gpu).detach()
            x = torch.cat([x, pad], dim=1)

        # --- Intra-block self-attention ---
        # (batch, block_num(m), seq_len(r), word_dim)
        x = torch.stack([x.narrow(1, i, r) for i in range(0, x.size(1), r)], dim=1)
        # (batch, block_num(m), seq_len(r), word_dim)
        h = self.mSA(x)
        # (batch, block_num(m), word_dim)
        v = self.s2tSA(h)

        # --- Inter-block self-attention ---
        # (batch, m, word_dim)
        o = self.mSA(v)
        # (batch, m, word_dim)
        G = F.sigmoid(self.g_W1(o) + self.g_W2(v) + self.g_b)
        # (batch, m, word_dim)
        e = G * o + (1 - G) * v

        # --- Context fusion ---
        # (batch, n, word_dim)
        E = torch.cat([torch.stack([e.select(1, i)] * r, dim=1) for i in range(e.size(1))], dim=1).narrow(1, 0, n)
        x = x.view(x.size(0), -1, x.size(-1)).narrow(1, 0, n)
        h = h.view(h.size(0), -1, h.size(-1)).narrow(1, 0, n)

        # (batch, n, word_dim * 3) -> (batch, n, word_dim)
        fusion = self.f_W1(torch.cat([x, h, E], dim=2))
        G = F.sigmoid(self.f_W2(torch.cat([x, h, E], dim=2)))
        # (batch, n, word_dim)
        u = G * fusion + (1 - G) * x

        return u


class s2tSA(customizedModule):
    def __init__(self, args, hidden_size):
        super(s2tSA, self).__init__()

        self.args = args
        self.s2t_W1 = self.customizedLinear(hidden_size, hidden_size, activation=nn.ReLU())
        self.s2t_W = self.customizedLinear(hidden_size, hidden_size)

    def forward(self, x):
        """
        source2token self-attention module
        :param x: (batch, (block_num), seq_len, hidden_size)
        :return: s: (batch, (block_num), hidden_size)
        """

        # (batch, (block_num), seq_len, word_dim)
        f = self.s2t_W1(x)
        f = F.softmax(self.s2t_W(f), dim=-2)
        # (batch, (block_num), word_dim)
        s = torch.sum(f * x, dim=-2)
        return s
