import pandas as pd
import os
import numpy as np
import string
from tokenizers.processors import TemplateProcessing
os.chdir(r'D:\data\COVID')

tokenizer_path=r'E:\classes\CS\CS5787\project\models\BPEtok\BPEtok.json'

df=pd.read_csv(r'true_ex_covid.csv')
df.head()

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers

tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
tokenizer.decoders = decoders.ByteLevel()
tokenizer.pad_token = '[PAD]'

tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
)

alph='abcdefghijklmnopqrstuvwxyz'+'*'
alph=[x for x in alph]

trainer = trainers.BpeTrainer(
    vocab_size=1000,
    initial_alphabet=alph,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]"],
)

data = df['CDR3'].tolist()+df['epitope'].tolist()


tokenizer.train_from_iterator(data, trainer=trainer)

tokenizer.save(
    path = tokenizer_path,
)
# =============================================================================
# tokenizer.save_pretrained(
#     path = tokenizer_path,
# )
# =============================================================================

from transformers import PreTrainedTokenizerFast
tokenizer2 = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

tokenizer2.pad_token = '[PAD]'
tokenizer2.encode(df.values.tolist()[0])
tokenizer2.batch_encode_plus(df.values.tolist()[:10],padding='longest')



