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


device = 'cuda' if torch.cuda.is_available() else 'cpu'


os.chdir(r'D:\data\COVID')



#df.head()


vocab_size=1000


np.random.seed(2021)


df=pd.read_csv(r'true_ex_covid.csv')
N=df.shape[0]

df=df.sample(frac=1, random_state=2021)

dftrain=df.iloc[:int(N*0.8),:]
dfvalid=df.iloc[int(N*0.8):int(N*0.9),:]
dftest=df.iloc[int(N*0.9):,:]


tokenizer_path=r'E:\classes\CS\CS5787\project\models\BPEtok\BPEtok.json'
tokenizer2 = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
tokenizer2.pad_token = '[PAD]'

#tokenized_dataset=tokenizer2.batch_encode_plus(df.values.tolist(),padding='longest')
#with open(r'E:\classes\CS\CS5787\project\data\tokenized_dataset.pkl','wb') as f:
#    pickle.dump(tokenized_dataset,f)
    
    
#with open(r'E:\classes\CS\CS5787\project\data\tokenized_dataset.pkl','rb') as f:
#    tokenized_dataset=pickle.load(f)
#N=len(tokenized_dataset['input_ids'])
    




class BatchSampler(Sampler):
    def __init__(self, tokenizer, data, neg_pos_ratio=1, batch_size=64):

        
        self.tokenizer = tokenizer

        self.data = data
        
        self.data['target']=1
                
        self.negN = int(len(self.data)*neg_pos_ratio)
        
        self.negdata=pd.concat((df.iloc[:,0].sample(self.negN, random_state=2021, replace=True).reset_index(drop=True),
                                df.iloc[:,1].sample(self.negN, random_state=2021, replace=True).reset_index(drop=True)),axis=1)
        self.negdata['target']=0
        
        self.data = pd.concat((self.data,self.negdata),axis=0).sample(frac=1, random_state=2021)

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
        inputs=self.tokenizer.batch_encode_plus(batch.iloc[:,:2].values.tolist(), padding='longest')
    
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        
        return {
                'input_ids': torch.tensor(ids, dtype=torch.long).squeeze().to(device),
                'mask': torch.tensor(mask, dtype=torch.long).squeeze().to(device),
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




configuration = BertConfig(vocab_size=vocab_size, hidden_size=768,
num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072,
hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02,
layer_norm_eps=1e-12, pad_token_id=0, gradient_checkpointing=False,
position_embedding_type='absolute', use_cache=True)



class BERT_dense(torch.nn.Module):
    def __init__(self):
        super(BERT_dense, self).__init__()
        self.bert = BertModel(configuration)
        self.l1 = nn.Linear(768, 256)
        self.l2 = nn.Linear(256, 1)
        #self.l3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):

        output = self.bert(inputs['input_ids'], attention_mask = inputs['mask'], token_type_ids = inputs['token_type_ids'])
        
        output = self.relu(self.l1(output[1]))
        output = self.l2(output)
        output = self.sigmoid(output)

        return output




def train(epoch, model, loader, optimizer, criterion):
    model.train()
    for _,data in enumerate(loader, 0):
        
        optimizer.zero_grad()
        outputs = model(data)
        target=data['target'].to(device)
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
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            outputs = model(data)
            target=data['target']
        
            actuals.extend(target)
            predictions.extend(outputs)
            
    return actuals, predictions
    
training_loader=BatchSampler(tokenizer2,dftrain, neg_pos_ratio=2)
val_loader=BatchSampler(tokenizer2,dfvalid, neg_pos_ratio=2)
test_loader=BatchSampler(tokenizer2,dftest, neg_pos_ratio=2)
    
    
#run_name='_'.join([str(x) for x in neurons])+'_fc'+str(fc_units)+'_lr{}_decay{}_adamw_drop{}_shuffletrain'.format(str(config_wandb.LEARNING_RATE),str(wdecay),str(dropout))

wandb.init(project="TCR_BERT", name='2_256_neg_pos2', reinit=True)

config = wandb.config
config.TRAIN_BATCH_SIZE = 64
config.TRAIN_EPOCHS = 50
config.LEARNING_RATE = 1e-4
config.SEED = 42

torch.manual_seed(config.SEED) # pytorch random seed
np.random.seed(config.SEED) # numpy random seed
torch.backends.cudnn.deterministic = True




model = BERT_dense()
model = model.to(device)

optimizer = torch.optim.Adam(params =  model.parameters(), lr=config.LEARNING_RATE)

criterion = nn.BCELoss()

wandb.watch(model, log="all")

best_val_loss=np.inf
for epoch in range(config.TRAIN_EPOCHS):
    train(epoch, model, training_loader, optimizer, criterion)
    acts,preds=validate(model, val_loader)
    acts,preds=torch.tensor(acts).to(device),torch.tensor(preds).to(device)
    val_loss=criterion(preds,acts).item()
    val_acc=np.mean((1*(preds>0.5)==acts).cpu().numpy())
    
    print('Epoch {} validation loss: {}, acc: {}'.format(str(epoch),str(val_loss),str(val_acc)))
    wandb.log({"validation loss": val_loss})
    wandb.log({"validation acc": val_acc})
    
    if best_val_loss>val_loss:
        best_val_loss=val_loss

    if epoch%10==0:
        acts,preds=validate(model, test_loader)
        acts,preds=torch.tensor(acts).to(device),torch.tensor(preds).to(device)
        test_loss=criterion(preds,acts).cpu().numpy()
        test_acc=np.mean((1*(preds>0.5)==acts).cpu().numpy())

        print('Epoch {} test loss: {}, acc: {}'.format(str(epoch),str(test_loss),str(test_acc)))
        wandb.log({"test loss": test_loss})
        wandb.log({"test acc": test_acc})            
