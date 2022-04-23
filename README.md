# Smiles LSTM

With fine-tuning

1. Train on ChEMBL 28
2. Fine-tune on user-provided set of molecules, or
3. Bias the generation in a goal-oriented fashion via policy gradient RL.

Draws from this [GitHub repo](https://github.com/bayeslabs/genmol/tree/Sunita/genmol) by BayesLabs
and the
associated [Medium post](https://medium.com/@sunitachoudhary103/generating-molecules-using-a-char-rnn-in-pytorch-16885fd9394b)
,  
this [blog post](https://www.cheminformania.com/master-your-molecule-generator-seq2seq-rnn-models-with-smiles-in-keras/)
by Esben Jannik Bjerrum and the [ReLeaSE](https://www.science.org/doi/10.1126/sciadv.aap7885) algorihm by Popova et al.

# 1. Train the prior model
```console
cd SmilesLSTM/prior
python train_prior.py
```

# 2. Finetune the prior model
Finetune the model onto a ChEMBL dump of compounds tested against A2aR
```console
python model/finetune.py \
  -p SmilesLSTM/prior/Smiles-LSTM_ChEMBL28_prior.pt \
  -f SmilesLSTM/input/ChEMBL_ADORA2a_IC50-Ki.csv.gz \
  -op finetuned \
  --smiles_col Smiles
```
