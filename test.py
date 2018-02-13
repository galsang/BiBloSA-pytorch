import argparse

import torch
from torch import nn

from model.model import NN4SNLI
from model.data import SNLI


def test(model, data, mode='test'):
    if mode == 'dev':
        iterator = iter(data.dev_iter)
    else:
        iterator = iter(data.test_iter)

    criterion = nn.CrossEntropyLoss()
    model.eval()
    acc, loss, size = 0, 0, 0

    for batch in iterator:
        pred = model(batch)

        batch_loss = criterion(pred, batch.label)
        loss += batch_loss.data[0]

        _, pred = pred.max(dim=1)
        acc += (pred == batch.label).sum().float()
        size += len(pred)

    acc /= size
    acc = acc.cpu().data[0]
    return loss, acc


def load_model(args, data):
    model = NN4SNLI(args, data)
    model.load_state_dict(torch.load(args.model_path))

    if args.gpu > -1:
        model.cuda(args.gpu)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--block-size', default=3, type=int)
    parser.add_argument('--data-type', default='SNLI')
    parser.add_argument('--dropout', default=0.25, type=float)
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    #parser.add_argument('--hidden-size', default=300, type=int)
    parser.add_argument('--mSA-scalar', default=5.0, type=float)
    parser.add_argument('--print-freq', default=2000, type=int)
    parser.add_argument('--word-dim', default=300, type=int)

    parser.add_argument('--model-path', required=True)

    args = parser.parse_args()

    print('loading SNLI data...')
    data = SNLI(args)

    setattr(args, 'word_vocab_size', len(data.TEXT.vocab))
    setattr(args, 'class_size', len(data.LABEL.vocab))

    print('loading model...')
    model = load_model(args, data)

    _, acc = test(model, data)

    print(f'test acc: {acc:.3f}')