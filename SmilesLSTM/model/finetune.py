import numpy as np
import pandas as pd
import torch

import model
import sample
import train
import utils

import json

logger = utils.logger

# Check if GPU is available
train_on_gpu = torch.cuda.is_available()
device = 'cuda' if train_on_gpu else 'cpu'

logger.info(f'Running on {device}')

# --------------- SETUP -----------------------------
with open('../net_config.json') as handle:
    config = json.load(handle)
logger.info(f'Config {config}')

# --------------- FINE TUNING -----------------------------
# Setup model
logger.info('Loading the unbiased model for finetuning')
net = model.CharRNN(utils.chars, n_hidden=config['n_hidden'], n_layers=config['n_layers'])
net.load_state_dict(torch.load('../prior/Smiles-LSTM_ChEMBL28_prior.pt', map_location=torch.device(device)))
print(net)

# Load training data
logger.info('Loading and processing input data for finetuning')
data = pd.read_csv('../input/ChEMBL_ADORA2a_IC50-Ki.csv.gz')
data = data[data['pChEMBL Value'] >= 7]

# Encode the text
actives = '\n'.join(data.Smiles)
encoded = np.array([utils.char2int[ch] for ch in actives])

# Train
logger.info('Finetuning')
train_info = train.train(
    net, encoded,
    epochs=config['n_epochs_finetune'],
    batch_size=config['batch_size'],
    seq_length=config['seq_length'],
    lr=config['lr'],
    print_every=10000
)
train_info = pd.DataFrame(train_info, columns=['epoch', 'step', 'train_loss', 'val_loss'])
train_info.to_csv('../output/Smiles-LSTM_ChEMBL28_finetune_info.csv', index=False)

# Sample model
logger.info('Sampling the finetuned model')
sample_ft = sample.get_sample_frame(net, size=100000, prime='C')
sample_ft['set'] = 'finetune'

# Save prior model and sample output
logger.info('Saving the finetuned model and its sample output')
torch.save(net.state_dict(), '../output/Smiles-LSTM_ChEMBL28_finetune.pt')
sample_ft.drop(columns=['ROMol']).to_csv('../output/Smiles-LSTM_ChEMBL28_finetune.csv')
