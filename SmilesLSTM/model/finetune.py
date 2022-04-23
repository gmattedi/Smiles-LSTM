import json
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from SmilesLSTM.model import model, sample, train, utils

logger = utils.logger


def finetune(
        net: model.CharRNN,
        smiles_lst: Sequence[str],
        output_prefix: str,
        epochs: int,
        batch_size: int,
        seq_length: int,
        size: int,
        lr: float
):
    smiles_concat = '\n'.join(smiles_lst)
    encoded = np.array([utils.char2int[ch] for ch in smiles_concat])

    # Train
    logger.info('Finetuning')
    train_info = train.train(
        net, encoded,
        epochs=epochs,
        batch_size=batch_size,
        seq_length=seq_length,
        lr=lr,
        log_every=10000
    )
    train_info = pd.DataFrame(train_info, columns=['epoch', 'step', 'train_loss', 'val_loss'])
    train_info.to_csv(output_prefix + '_info.csv', index=False)

    # Sample model
    logger.info('Sampling the finetuned model')
    sample_ft = sample.get_sample_frame(net, size=size, prime='C')
    sample_ft['set'] = 'finetune'

    # Save prior model and sample output
    logger.info('Saving the finetuned model and its sample output')
    torch.save(net.state_dict(), output_prefix + '.pt')
    sample_ft.drop(columns=['ROMol']).to_csv(output_prefix + '.csv')


if __name__ == '__main__':
    import argparse

    with open(utils.MODULE_PATH + '/net_config.json') as handle:
        config = json.load(handle)

    parser = argparse.ArgumentParser(description='Finetune SmilesLSTM model')
    parser.add_argument('-p', '--params', help='Trained model parameters (.pt file)', required=True)
    parser.add_argument('--hidden', help='Hidden units (default: %(default)d)', required=False,
                        default=config['n_hidden'], type=int)
    parser.add_argument('--layers', help='Layers (default: %(default)d)', required=False, default=config['n_layers'],
                        type=int)
    parser.add_argument('--batch_size', help='Batch size (default: %(default)d)', required=False,
                        default=config['batch_size'],
                        type=int)
    parser.add_argument('--epochs', help='Epochs (default: %(default)d)', required=False,
                        default=config['n_epochs_finetune'],
                        type=int)
    parser.add_argument('--seq_length', help='Number of characters for minibatch (default: %(default)d)',
                        required=False,
                        default=config['seq_length'],
                        type=int)
    parser.add_argument('-lr', '--learning_rate', help='Learning rate (default: %(default)f)', required=False,
                        default=config['lr'],
                        type=float)
    parser.add_argument('-f', '--finetune_csv', help='CSV/CSV.GZ file of SMILES strings for finetuning', required=True,
                        type=str)
    parser.add_argument('--smiles_col', help='SMILES column (default: $(default)s)', required=False, type=str,
                        default='SMILES')
    parser.add_argument('-s', '--size', help='Sample this many characters (default: %(default)d)', required=False,
                        default=100000,
                        type=int)
    parser.add_argument('-op', '--output_prefix', help='Prefix for output files', required=True, type=str)
    args = parser.parse_args()

    train_on_gpu = torch.cuda.is_available()
    device = 'cuda' if train_on_gpu else 'cpu'

    logger.info('Loading the model')
    net = model.CharRNN(utils.chars, n_hidden=args.hidden, n_layers=args.layers)
    net.load_state_dict(torch.load(args.params, map_location=torch.device(device)))

    logger.info('Loading and processing input data for finetuning')
    data = pd.read_csv(args.finetune_csv)
    smiles = data[args.smiles_col].values

    finetune(
        net,
        smiles_lst=smiles,
        output_prefix=args.output_prefix,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        size=args.size,
        lr=args.learning_rate
    )
