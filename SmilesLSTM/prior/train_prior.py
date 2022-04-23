import json

import numpy as np
import pandas as pd
import torch

from SmilesLSTM.model import model, sample, train, utils

logger = utils.logger

# Check if GPU is available
train_on_gpu = torch.cuda.is_available()
device = 'cuda' if train_on_gpu else 'cpu'

logger.info(f'Running on {device}')

# --------------- SETUP -----------------------------
with open(utils.MODULE_PATH + '/net_config.json') as handle:
    config = json.load(handle)
logger.info(f'Config {config}')

# --------------- PRIOR MODEL -----------------------------
# Setup model
logger.info('Instantiating the model')
net = model.CharRNN(utils.chars, n_hidden=config['n_hidden'], n_layers=config['n_layers'])
logger.info(net)

# Load training data
logger.info('Loading and processing input data')
chemreps = pd.read_csv(utils.MODULE_PATH + '/input/chembl_28_chemreps.csv.gz')
chemreps = chemreps[chemreps.canonical_smiles.str.len() <= 100]

# Encode the text
text = '\n'.join(chemreps.canonical_smiles.values)
encoded = np.array([utils.char2int[ch] for ch in text])

# Train
logger.info('Training')
train_info = train.train(
    net, encoded,
    epochs=config['n_epochs'],
    batch_size=config['batch_size'],
    seq_length=config['seq_length'],
    lr=config['lr'],
    log_every=10000
)
train_info = pd.DataFrame(train_info, columns=['epoch', 'step', 'train_loss', 'val_loss'])
train_info.to_csv(utils.MODULE_PATH + '/prior/Smiles-LSTM_ChEMBL28_prior_info.csv', index=False)

# Sample model
logger.info('Sampling the unbiased model')
sample_prior = sample.get_sample_frame(net, size=100000, prime='C')
sample_prior['set'] = 'prior'

# Save prior model and sample output
logger.info('Saving the prior model and its sample output')
torch.save(net.state_dict(), utils.MODULE_PATH + '/prior/Smiles-LSTM_ChEMBL28_prior.pt')
sample_prior.drop(columns=['ROMol']).to_csv(utils.MODULE_PATH + '/prior/Smiles-LSTM_ChEMBL28_prior.csv')
