import pandas as pd
import os
import numpy as np
import torch
import wandb
from torch import nn
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import PreTrainedTokenizerFast
from transformers import BertModel, BertConfig
import pickle
import torch.autograd as autograd

from sklearn.metrics import roc_auc_score, confusion_matrix, average_precision_score


batch_size=150
fc_neurons=[]

nlayers=4
hidden=256
#nlayers=12
#hidden=768


dropout=0.0

pretrainedpath=r'E:\classes\CS\CS5787\project\results\BERT mini\50.pt'
#pretrainedpath='/N/u/astrelt/Carbonate/Documents/CS5787_project/models/VJCOVID_BERT_MLM_FIXEDSPLIT_h768_nl12_d0.0_lr0.0001_decay0.01_neg_pos5_spawn5/3.pt'

#continuefrompath='/N/u/astrelt/Carbonate/Documents/CS5787_project/models/BERT_pred_pretrained_h768_nl12_d0.0_lr1e-05_decay1e-05_neg_pos5/13.pt'




device = 'cuda' if torch.cuda.is_available() else 'cpu'

#os.chdir(r'/N/u/astrelt/Carbonate/Documents/CS5787_project/data')

os.chdir(r'D:\data\COVID')

amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
amino_to_ix = {amino: index for index, amino in enumerate(['[PAD]'] + amino_acids)}
vocab_size=len(amino_to_ix)
amino_to_ix['[UNK]']=vocab_size

amino_acids2 = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
amino_to_ix2 = {amino: len(amino_to_ix)+index for index, amino in enumerate(['[PAD]'] + amino_acids2)}
vocab_size2=len(amino_to_ix2)
amino_to_ix2['[UNK]']=amino_to_ix['[UNK]']
amino_to_ix2['[PAD]']=amino_to_ix['[PAD]']
amino_to_ix2['[SEP]']=amino_to_ix['[SEP]']=np.max(list(amino_to_ix2.values()))+1
amino_to_ix2['[CLS]']=amino_to_ix['[CLS]']=np.max(list(amino_to_ix2.values()))+1
amino_to_ix2['[MASK]']=amino_to_ix['[MASK]']=np.max(list(amino_to_ix2.values()))+1

nclasses=np.max(list(amino_to_ix2.values()))+1

np.random.seed(2021)

tokenizer=np.vectorize(lambda x: amino_to_ix.get(x,amino_to_ix['[UNK]']))
tokenizer2=np.vectorize(lambda x: amino_to_ix2.get(x,amino_to_ix2['[UNK]']))

mask_id=amino_to_ix2['[MASK]']
sep_id=amino_to_ix2['[SEP]']
cls_id=amino_to_ix2['[CLS]']

df=pd.read_csv(r'test_1x1.csv')




#WANT TO add sep token between and at the end, and run a separate dict for epitope
#token type id is 0s for TCRs including SEP, then 1s for the other seq



def pad_seq(a,maxlen):
    delta=maxlen-a.shape[0]
    if delta<=0:
        return a
    return np.pad(a,pad_width=(0,delta),constant_values=(0, 0))


class BatchSampler(Sampler):
    def __init__(self, data, tokenizer1=tokenizer, tokenizer2=tokenizer2, batch_size=64):

        
        self.tokenizer = tokenizer
        self.tokenizer2 = tokenizer2

        self.data = data
        
        
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
    
        inputs1=[self.tokenizer(['[CLS]']+[x for x in y]+['[SEP]']).tolist() for y in batch.iloc[:,0].values.tolist()]
        lens1=[len(x) for x in inputs1]
        
        inputs2=[self.tokenizer2([x for x in y]+['[SEP]']).tolist() for y in batch.iloc[:,1].values.tolist()]
        lens2=[len(x) for x in inputs2]
        
        maxlen=np.max(np.array(lens1)+np.array(lens2))
        
        inputs=[inputs1[i]+inputs2[i] for i in range(len(inputs1))]
            
        inputs=np.stack([pad_seq(np.array(x),maxlen) for x in inputs])
        
        attention_mask=1*(inputs!=self.tokenizer('[PAD]'))
        
        token_type_ids=np.ones_like(inputs)
        
        for row_id in range(token_type_ids.shape[0]):
            token_type_ids[row_id,:lens1[row_id]]=0
        
        
    
        return {#'source': batch['source'].values,
                'input_ids': torch.tensor(inputs, dtype=torch.long).squeeze().to(device),
                'mask': torch.tensor(attention_mask, dtype=torch.long).squeeze().to(device),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long).squeeze().to(device),
                'target': torch.tensor(batch['target'].tolist(), dtype=torch.float).to(device)
            }
    
    
    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i in self.batches:
            yield i       





configuration = BertConfig(vocab_size=np.max(list(amino_to_ix2.values()))+1, hidden_size=768,
num_hidden_layers=nlayers, num_attention_heads=int(hidden/64), intermediate_size=int(4*hidden),
hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02,
layer_norm_eps=1e-12, pad_token_id=0, gradient_checkpointing=False,
position_embedding_type='absolute', use_cache=True)

#BATCHNORM?

class BERT_dense(torch.nn.Module):
    def __init__(self, fc_neurons=[],dropout=0):
        super(BERT_dense, self).__init__()
        self.bert = BertModel(configuration)
        self.relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
        self.dropoutlayers=nn.ModuleList()
        self.neurons=fc_neurons
        
        current_dim = 768
        self.layers = nn.ModuleList()
        for n in fc_neurons:
             self.layers.append(nn.Linear(current_dim, n))
             if dropout>0:
               self.dropoutlayers.append(nn.Dropout(dropout))
             current_dim = n
        self.layers.append(nn.Linear(current_dim, 1))

        
    def forward(self, inputs):

        x = self.bert(inputs['input_ids'], attention_mask = inputs['mask'], token_type_ids = inputs['token_type_ids'])
        x=x[0][:,0,:]
        for c,layer in enumerate(self.layers[:-1]):
             x = self.relu(layer(x))
             if self.dropout>0:
               x=self.dropoutlayers[c](x)
        out = self.sigmoid(self.layers[-1](x))
        return out 



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
   # sources = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            outputs = model(data)
            target=data['target']
        
            actuals.extend(target)
            predictions.extend(outputs)
            #sources.extend(data['source'])
            
    return actuals, predictions
    



test_loader=BatchSampler(df, batch_size=batch_size)
    
    
#run_name='_'.join([str(x) for x in neurons])+'_fc'+str(fc_units)+'_lr{}_decay{}_adamw_drop{}_shuffletrain'.format(str(config_wandb.LEARNING_RATE),str(wdecay),str(dropout))






model = BERT_dense(fc_neurons=fc_neurons, dropout=dropout).to(device)


checkpoint = torch.load(pretrainedpath)

model.load_state_dict(checkpoint['model_state_dict'])






acts,preds=validate(model, test_loader)
acts,preds=torch.tensor(acts).to(device),torch.tensor(preds).to(device)

df['pred_conf']=preds.cpu().numpy()

df.to_csv(r'E:\classes\CS\CS5787\project\results\BERT mini\preds.csv',index=False)
