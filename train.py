import argparse
import copy
import os
import torch

from torch import nn, optim
from tensorboardX import SummaryWriter
from time import gmtime, strftime

from model.model import NN4SNLI
from model.data import SNLI
from test import test


def train(args, data):
    model = NN4SNLI(args, data)
    if args.gpu > -1:
        model.cuda(args.gpu)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir='runs/' + args.model_time)

    model.train()
    loss, last_epoch = 0, -1
    max_dev_acc, max_test_acc = 0, 0

    iterator = data.train_iter
    for i, batch in enumerate(iterator):
        present_epoch = int(iterator.epoch)
        if present_epoch == args.epoch:
            break
        if present_epoch > last_epoch:
            print('epoch:', present_epoch + 1)
        last_epoch = present_epoch

        pred = model(batch)

        optimizer.zero_grad()
        batch_loss = criterion(pred, batch.label)
        loss += batch_loss.data[0]
        batch_loss.backward()
        optimizer.step()

        if (i + 1) % args.print_freq == 0:
            dev_loss, dev_acc = test(model, data, mode='dev')
            test_loss, test_acc = test(model, data)
            c = (i + 1) // args.print_freq

            writer.add_scalar('loss/train', loss, c)
            writer.add_scalar('loss/dev', dev_loss, c)
            writer.add_scalar('acc/dev', dev_acc, c)
            writer.add_scalar('loss/test', test_loss, c)
            writer.add_scalar('acc/test', test_acc, c)

            print(f'train loss: {loss:.3f} / dev loss: {dev_loss:.3f} / test loss: {test_loss:.3f}'
                  f' / dev acc: {dev_acc:.3f} / test acc: {test_acc:.3f}')

            if dev_acc > max_dev_acc:
                max_dev_acc = dev_acc
                max_test_acc = test_acc
                best_model = copy.deepcopy(model)

            loss = 0
            model.train()

    writer.close()
    print(f'max dev acc: {max_dev_acc:.3f} / max test acc: {max_test_acc:.3f}')

    return best_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--block-size', default=3, type=int)
    parser.add_argument('--data-type', default='SNLI')
    parser.add_argument('--dropout', default=0.25, type=float)
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    #parser.add_argument('--hidden-size', default=300, type=int)
    parser.add_argument('--learning-rate', default=0.5, type=float)
    parser.add_argument('--mSA-scalar', default=5.0, type=float)
    parser.add_argument('--print-freq', default=2000, type=int)
    parser.add_argument('--weight-decay', default=5e-5, type=float)
    parser.add_argument('--word-dim', default=300, type=int)

    args = parser.parse_args()

    print('loading SNLI data...')
    data = SNLI(args)

    setattr(args, 'word_vocab_size', len(data.TEXT.vocab))
    setattr(args, 'class_size', len(data.LABEL.vocab))
    setattr(args, 'model_time', strftime('%H:%M:%S', gmtime()))
    # if block size is lower than 0, a heuristic for block size is applied.
    if args.block_size < 0:
        args.block_size = data.block_size

    print('training start!')
    best_model = train(args, data)

    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    torch.save(best_model.state_dict(), f'saved_models/BiBloSA_{args.data_type}_{args.model_time}.pt')

    print('training finished!')


if __name__ == '__main__':
    main()
