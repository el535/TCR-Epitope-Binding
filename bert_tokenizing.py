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

from sklearn.metrics import roc_auc_score

projectname='BERT_pred_pretrained'
neg_pos_ratio=5
batch_size=150
fc_neurons=[]

nlayers=4
hidden=256
#nlayers=12
#hidden=768


dropout=0.0
lr=1e-6
wdecay=1e-5
pos_weight=neg_pos_ratio/(neg_pos_ratio+1)
neg_weight=1-pos_weight
epochs=100

pretrainedpath='E:/classes/CS/CS5787/project/models/VJCOVID_BERT_MLM_FIXEDSPLIT_h256_nl4_d0.0_lr0.0001_decay0.01_neg_pos5_spawn5/7.pt'


runname='h{}'.format(str(hidden))+'_'+'nl'+str(nlayers)+'_'+''.join([str(x) for x in fc_neurons])+'d{}_lr{}_decay{}_neg_pos{}'.format(
    str(dropout),str(lr),str(wdecay),str(neg_pos_ratio))
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

savein=os.path.join(r'/N/u/astrelt/Carbonate/Documents/CS5787_project/models',projectname+'_'+runname)

if not os.path.exists(savein):
	os.mkdir(savein)

os.chdir(r'/N/u/astrelt/Carbonate/Documents/CS5787_project/data')
#os.chdir(r'D:\data\COVID')

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




#WANT TO add sep token between and at the end, and run a separate dict for epitope
#token type id is 0s for TCRs including SEP, then 1s for the other seq



def pad_seq(a,maxlen):
    delta=maxlen-a.shape[0]
    if delta<=0:
        return a
    return np.pad(a,pad_width=(0,delta),constant_values=(0, 0))


class BatchSampler(Sampler):
    def __init__(self, data, tokenizer1=tokenizer, tokenizer2=tokenizer2, neg_pos_ratio=1, batch_size=64):

        
        self.tokenizer = tokenizer
        self.tokenizer2 = tokenizer2

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
        
        
    
        return {'source': batch['source'].values,
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


