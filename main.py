import numpy as np
import pandas as pd
import torch

import model
import sample
import train
import utils

logger = utils.logger

# Check if GPU is available
train_on_gpu = torch.cuda.is_available()
device = 'cuda' if train_on_gpu else 'cpu'

logger.info(f'Running on {device}')

# --------------- SETUP -----------------------------
config = {
    'n_hidden': 56,
    'n_layers': 2,
    'batch_size': 32,
    'seq_length': 50,
    'n_epochs': 10,
    'n_epochs_finetune': 50,
    'lr': 0.001
}
logger.info(f'Config {config}')

# --------------- PRIOR MODEL -----------------------------
# Setup model
logger.info('Instantiating the model')
net = model.CharRNN(utils.chars, n_hidden=config['n_hidden'], n_layers=config['n_layers'])
logger.info(net)

# Load training data
logger.info('Loading and processing input data')
chemreps = pd.read_csv("input/chembl_28_chemreps.csv.gz")
chemreps = chemreps[chemreps.canonical_smiles.str.len() <= 100]

# Encode the text
text = '\n'.join(chemreps.canonical_smiles.values)
encoded = np.array([utils.char2int[ch] for ch in text])

# Train
logger.info('Training')
train.train(
    net, encoded,
    epochs=config['n_epochs'],
    batch_size=config['batch_size'],
    seq_length=config['seq_length'],
    lr=config['lr'],
    print_every=10000
)

# Sample model
logger.info('Sampling the unbiased model')
sample_prior = sample.get_sample_frame(net, size=100000, prime='B')
sample_prior['set'] = 'prior'

# Save prior model and sample output
logger.info('Saving the unbiased model and its sample output')
torch.save(net.state_dict(), 'output/Smiles-LSTM_ChEMBL28_prior.pt')
sample_prior.drop(columns=['ROMol']).to_csv('output/Smiles-LSTM_ChEMBL28_prior.csv')

# --------------- FINE TUNING -----------------------------
# Setup model
logger.info('Reloading the unbiased model for finetuning')
net = model.CharRNN(utils.chars, n_hidden=config['n_hidden'], n_layers=config['n_layers'])
net.load_state_dict(torch.load('output/Smiles-LSTM_ChEMBL28_prior.pt', map_location=torch.device(device)))
print(net)

# Load training data
logger.info('Loading and processing input data for finetuning')
data = pd.read_csv('input/chembl-adora2a-ic50-ki/ChEMBL_ADORA2a_IC50-Ki.csv')
data = data[data['pChEMBL Value'] >= 7]

# Encode the text
actives = '\n'.join(data.Smiles)
encoded = np.array([utils.char2int[ch] for ch in actives])

# Train
logger.info('Finetuning')
train.train(
    net, encoded,
    epochs=config['n_epochs_finetune'],
    batch_size=config['batch_size'],
    seq_length=config['seq_length'],
    lr=config['lr'],
    print_every=10000
)

# Sample model
logger.info('Sampling the finetuned model')
sample_ft = sample.get_sample_frame(net, size=100000, prime='B')
sample_ft['set'] = 'finetune'

# Save prior model and sample output
logger.info('Saving the finetuned model and its sample output')
torch.save(net.state_dict(), 'output/Smiles-LSTM_ChEMBL28_finetune.pt')
sample_prior.drop(columns=['ROMol']).to_csv('output/Smiles-LSTM_ChEMBL28_finetune.csv')

# Combine samples from prior and fine-tuned model and save
sample_both = pd.concat([sample_prior, sample_ft])
sample_both.drop(columns=['ROMol']).to_csv('output/Smiles-LSTM_ChEMBL28_both.csv')
