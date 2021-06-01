import pandas as pd
import os
import numpy as np
import torch
import wandb
from torch import nn
from torch.utils.data import Dataset, DataLoader, Sampler
#from transformers import PreTrainedTokenizerFast
#from transformers import BertModel, BertConfig
import pickle
import torch.autograd as autograd

from sklearn.metrics import roc_auc_score


projectname='VJCOV_LSTM2_FIXEDSPLIT'
neg_pos_ratio=5
batch_size=150
embedding_dim=10    
lstm_dim=500
dropout=0.1
lr=1e-3
wdecay=1e-5
pos_weight=neg_pos_ratio/(neg_pos_ratio+1)
neg_weight=1-pos_weight
epochs=100


runname='e{}_l{}_d{}_lr{}_decay{}_neg_pos{}'.format(
    str(embedding_dim),str(lstm_dim),str(dropout),str(lr),str(wdecay),str(neg_pos_ratio))
    


device = 'cuda' if torch.cuda.is_available() else 'cpu'

savein=os.path.join(r'/N/u/astrelt/Carbonate/Documents/CS5787_project/models',projectname+'_'+runname)

if not os.path.exists(savein):
	os.mkdir(savein)
	
os.chdir(r'/N/u/astrelt/Carbonate/Documents/CS5787_project/data')

#os.chdir(r'D:\data\COVID')

amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}


vocab_size=len(amino_to_ix)
amino_to_ix['[UNK]']=vocab_size

np.random.seed(2021)

tokenizer=np.vectorize(lambda x: amino_to_ix.get(x,amino_to_ix['[UNK]']))

df=pd.read_csv(r'dat_train.csv')
df=df.loc[df['CDR3']!='unproductive']
df['remove']=df.apply(lambda row: ('*' in row['CDR3'])|('*' in row['epitope']),axis=1)
df=df.loc[df['remove']!=True]

dftest=pd.read_csv(r'dat_test.csv')
dftest=dftest.loc[dftest['CDR3']!='unproductive']
dftest['remove']=dftest.apply(lambda row: ('*' in row['CDR3'])|('*' in row['epitope']),axis=1)
dftest=dftest.loc[dftest['remove']!=True]

N=df.shape[0]


df=df.sample(frac=1, random_state=2021)
df['CDR3_ep']=df['CDR3']+'_'+df['epitope']
binding_seqs=df['CDR3_ep'].tolist()

dftest['CDR3_ep']=dftest['CDR3']+'_'+dftest['epitope']
binding_seqs=dftest['CDR3_ep'].tolist()+binding_seqs


dftrain=df.iloc[:int(N*0.9),:]
dfvalid=df.iloc[int(N*0.9):,:]




def pad_seq(a,maxlen):
    delta=maxlen-a.shape[0]
    if delta<=0:
        return a
    return np.pad(a,pad_width=(0,delta),constant_values=(0, 0))


class BatchSampler(Sampler):
    def __init__(self, data, tokenizer=tokenizer, neg_pos_ratio=1, batch_size=64):

        
        self.tokenizer = tokenizer

        self.data = data.copy()
        
        self.data['target']=1
                
        self.negN = int(len(self.data)*neg_pos_ratio)
        
        
        self.negdata=pd.concat((data.iloc[:,0].sample(self.negN, random_state=2021, replace=True).reset_index(drop=True),
                                data.iloc[:,1].sample(self.negN, random_state=2022, replace=True).reset_index(drop=True)),axis=1)
        self.negdata['target']=0
        self.negdata['CDR3_ep']=self.negdata['CDR3']+'_'+self.negdata['epitope']
        
        self.negdata=self.negdata.loc[~self.negdata['CDR3_ep'].isin(binding_seqs)]
        
        self.negdata2=pd.concat((data.iloc[:,0].sample(self.negN-self.negdata.shape[0], random_state=2023, replace=True).reset_index(drop=True),
                                data.iloc[:,1].sample(self.negN-self.negdata.shape[0], random_state=2024, replace=True).reset_index(drop=True)),axis=1)
        self.negdata2['target']=0
        self.negdata2['CDR3_ep']=self.negdata2['CDR3']+'_'+self.negdata2['epitope']
        
        self.negdata2=self.negdata2.loc[~self.negdata2['CDR3_ep'].isin(binding_seqs)]
       
        
        self.data = pd.concat((self.data,self.negdata,self.negdata2),axis=0).sample(frac=1, random_state=2023)

        self.batch_size = batch_size

        self.num_batches = int(np.ceil(len(self.data)/self.batch_size))
        
        self.batchify()
    
        self.tokenize()
        
    def batchify(self):
        self.batches=[]
        
        for i in range(self.num_batches-1):
            self.batches.append(self.data.iloc[i*self.batch_size:(i+1)*self.batch_size,:])
        self.batches.append(self.data.iloc[(i+1)*self.batch_size:,:])

        return
    
    def tokenize(self):
        
        for c,batch in enumerate(self.batches):
            self.batches[c]=self.tokenize_batch(batch)
        return
    
    
    def tokenize_batch(self, batch):
        inputs1=np.array([self.tokenizer([x for x in y]) for y in batch.iloc[:,0].values.tolist()])
        lens1=[x.shape[0] for x in inputs1]
        maxlen=np.max(lens1)
    
        inputs1=np.stack([pad_seq(x,maxlen) for x in inputs1])
    
        inputs2=np.array([self.tokenizer([x for x in y]) for y in batch.iloc[:,1].values.tolist()])
        lens2=[x.shape[0] for x in inputs2]
        maxlen=np.max(lens2)
    
        inputs2=np.stack([pad_seq(x,maxlen) for x in inputs2])    
    
        

        return {'source': batch['source'].values,
                'input1': torch.tensor(inputs1, dtype=torch.long).squeeze().to(device),
                'input2': torch.tensor(inputs2, dtype=torch.long).squeeze().to(device),
                'lens1': torch.tensor(lens1, dtype=torch.long).squeeze().to(device),
                'lens2': torch.tensor(lens2, dtype=torch.long).squeeze().to(device),
                'target': torch.tensor(batch['target'].tolist(), dtype=torch.float).to(device)
            }

    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i in self.batches:
            yield i    


class DoubleLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, lstm_dim, dropout, device):
        super(DoubleLSTMClassifier, self).__init__()
        # GPU
        self.device = device
        # Dimensions
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.dropout = dropout
        # Embedding matrices - 20 amino acids + padding
        self.tcr_embedding = nn.Embedding(len(amino_to_ix), embedding_dim, padding_idx=0)
        self.pep_embedding = nn.Embedding(len(amino_to_ix), embedding_dim, padding_idx=0)
        # RNN - LSTM
        self.tcr_lstm = nn.LSTM(embedding_dim, lstm_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.pep_lstm = nn.LSTM(embedding_dim, lstm_dim, num_layers=2, batch_first=True, dropout=dropout)
        # MLP
        self.hidden_layer = nn.Linear(lstm_dim * 2, lstm_dim)
        self.relu = torch.nn.LeakyReLU()
        self.output_layer = nn.Linear(lstm_dim, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid=nn.Sigmoid()
    def init_hidden(self, batch_size):
        return (torch.zeros(2, batch_size, self.lstm_dim, requires_grad=True).to(self.device),
                torch.zeros(2, batch_size, self.lstm_dim, requires_grad=True).to(self.device))

    def lstm_pass(self, lstm, padded_embeds):

        bs = padded_embeds.shape[0]
        hidden = self.init_hidden(bs)
        # Feed into the RNN
        lstm_out, hidden = lstm(padded_embeds, hidden)
        
        return lstm_out

    def forward(self, data):
        # TCR Encoder:
        # Embedding
        tcr_embeds = self.tcr_embedding(data['input1'])
        # LSTM Acceptor
        tcr_lstm_out = self.lstm_pass(self.tcr_lstm, tcr_embeds)
        tcr_last_cell = torch.cat([tcr_lstm_out[i, j.data - 1,:] for i, j in enumerate(data['lens1'])]).view(len(data['lens1']), self.lstm_dim)

        # PEPTIDE Encoder:
        # Embedding
        pep_embeds = self.pep_embedding(data['input2'])
        # LSTM Acceptor
        pep_lstm_out = self.lstm_pass(self.pep_lstm, pep_embeds)
        pep_last_cell = torch.cat([pep_lstm_out[i, j.data - 1] for i, j in enumerate(data['lens2'])]).view(len(data['lens2']), self.lstm_dim)

        # MLP Classifier
        tcr_pep_concat = torch.cat([tcr_last_cell, pep_last_cell], 1)
        hidden_output = self.dropout(self.relu(self.hidden_layer(tcr_pep_concat)))
        mlp_output = self.output_layer(hidden_output)
        output = self.sigmoid(mlp_output)
        return output




def train(epoch, model, loader, optimizer, criterion,pos_weight):
    model.train()
    for _,data in enumerate(loader, 0):
        
        optimizer.zero_grad()
        outputs = model(data)
        target=data['target'].to(device)
        weights=pos_weight*target+(1-pos_weight)*(1-target)
        criterion.weight = weights
        loss = criterion(outputs.squeeze(), target.squeeze())

        if _%10 == 0:
            wandb.log({"Training Loss": loss.item()})

        if _%500==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        loss.backward()
        optimizer.step()
        
    return


def validate(model, loader):
    model.eval()
    predictions = []
    actuals = []
    sources = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):

            outputs = model(data)
            target=data['target']
        
            actuals.extend(target)
            predictions.extend(outputs)
            sources.extend(data['source'])

    return actuals, predictions, sources
    



training_loader=BatchSampler(dftrain, neg_pos_ratio=neg_pos_ratio, batch_size=batch_size)
val_loader=BatchSampler(dfvalid, neg_pos_ratio=neg_pos_ratio, batch_size=batch_size)
test_loader=BatchSampler(dftest, neg_pos_ratio=neg_pos_ratio, batch_size=batch_size)
    
    
#run_name='_'.join([str(x) for x in neurons])+'_fc'+str(fc_units)+'_lr{}_decay{}_adamw_drop{}_shuffletrain'.format(str(config_wandb.LEARNING_RATE),str(wdecay),str(dropout))




wandb.init(project=projectname, name=runname, reinit=True)

config = wandb.config
config.TRAIN_BATCH_SIZE = batch_size
config.TRAIN_EPOCHS = epochs
config.LEARNING_RATE = lr
config.SEED = 42

torch.manual_seed(config.SEED) # pytorch random seed
np.random.seed(config.SEED) # numpy random seed
torch.backends.cudnn.deterministic = True



model = DoubleLSTMClassifier(embedding_dim=embedding_dim, lstm_dim=lstm_dim, dropout=dropout, device=device).to(device)


optimizer = torch.optim.AdamW(params =  model.parameters(), lr=config.LEARNING_RATE, weight_decay=wdecay)

criterion = nn.BCELoss()

#criterion = lambda output, target: torch.neg(torch.mean(pos_weight * (target * torch.log(output)) + \
#              neg_weight*((1 - target) * torch.log(1 - output))))
               
wandb.watch(model, log="all")

best_val_loss=np.inf
for epoch in range(config.TRAIN_EPOCHS):
    train(epoch, model, training_loader, optimizer, criterion, pos_weight)
    acts,preds,sources=validate(model, val_loader)
    acts,preds=torch.tensor(acts).to(device),torch.tensor(preds).to(device)
    weights=pos_weight*acts+(1-pos_weight)*(1-acts)
    criterion.weight = weights
    val_loss=criterion(preds,acts).item()
    val_acc=np.mean((1*(preds>0.5)==acts).cpu().numpy())
    val_auc=roc_auc_score(acts.cpu().numpy(), preds.cpu().numpy())
    print('Epoch {} validation loss: {}, acc: {}'.format(str(epoch),str(val_loss),str(val_acc)))
    wandb.log({"validation loss": val_loss})
    wandb.log({"validation acc": val_acc})
    wandb.log({"validation AUC": val_auc})
    
    cov_mask=np.array(sources)=='COVID_seqs'
    mcpas_mask=np.array(sources)=='McPAS_seqs'
    vdj_mask=np.array(sources)=='VDJDB_seqs'
    
    wandb.log({"validation recall COVID": np.mean((1*(preds[cov_mask]>0.5)==acts[cov_mask]).cpu().numpy())})
    
    wandb.log({"validation recall McPAS": np.mean((1*(preds[mcpas_mask]>0.5)==acts[mcpas_mask]).cpu().numpy())})
    
    wandb.log({"validation recall VDJDB": np.mean((1*(preds[vdj_mask]>0.5)==acts[vdj_mask]).cpu().numpy())})
    
    
    
    if best_val_loss>val_loss:
        best_val_loss=val_loss
        
        PATH=os.path.join(savein,'{}.pt'.format(str(epoch)))
        
        torch.save({'epoch':epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_loss}, PATH)

    if epoch%10==0:
        acts,preds,sources=validate(model, test_loader)
        acts,preds=torch.tensor(acts).to(device),torch.tensor(preds).to(device)
        weights=pos_weight*acts+(1-pos_weight)*(1-acts)
        criterion.weight = weights
        test_loss=criterion(preds,acts).cpu().numpy()
        test_acc=np.mean((1*(preds>0.5)==acts).cpu().numpy())
        test_auc=roc_auc_score(acts.cpu().numpy(), preds.cpu().numpy())
        print('Epoch {} test loss: {}, acc: {}'.format(str(epoch),str(test_loss),str(test_acc)))
        wandb.log({"test loss": test_loss})
        wandb.log({"test acc": test_acc})
        wandb.log({"test AUC": test_auc})

        cov_mask=np.array(sources)=='COVID_seqs'
        mcpas_mask=np.array(sources)=='McPAS_seqs'
        vdj_mask=np.array(sources)=='VDJDB_seqs'
        
        wandb.log({"test recall COVID": np.mean((1*(preds[cov_mask]>0.5)==acts[cov_mask]).cpu().numpy())})
        
        wandb.log({"test recall McPAS": np.mean((1*(preds[mcpas_mask]>0.5)==acts[mcpas_mask]).cpu().numpy())})
        
        wandb.log({"test recall VDJDB": np.mean((1*(preds[vdj_mask]>0.5)==acts[vdj_mask]).cpu().numpy())})
