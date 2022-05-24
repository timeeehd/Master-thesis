import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPTNeoForCausalLM, TrainingArguments, Trainer
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F
import csv
import os
from tqdm import tqdm
import sys

print('test')
# Set the random seed to a fixed value to get reproducible results 
torch.manual_seed(42)
print('test2')
# Download the pre-trained GPT-Neo model's tokenizer
# Add the custom tokens denoting the beginning and the end 
# of the sequence and a special token for padding
# tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B",    
#                             bos_token="<|startoftext|>",
#                             eos_token="<|endoftext|>",
#                             pad_token="<|pad|>")
#Get the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
print('initilize model')
# model.resize_token_embeddings(len(tokenizer))
# model.load_state_dict(torch.load('models/GPT2-med-2048-512.pt', map_location=torch.device('cpu')))
# model.load_state_dict(torch.load('models/GPT2-small.pt'))

print('add special tokens')
special_tokens_dict = {
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "pad_token": "<PAD>",
        "additional_special_tokens": [
            "<endprompt>",
        ],
    }
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print('initialize model')
model = GPT2LMHeadModel.from_pretrained("gpt2")
print('resize model')

model.resize_token_embeddings(len(tokenizer))
# num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
# Download the pre-trained GPT-Neo model and transfer it to the GPU

# model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").cuda()
# Resize the token embeddings because we've just added 3 new tokens 


with open('data/Fairy_tales_combined (1).txt', "r", encoding='utf-8-sig') as file:
    data = file.readlines()
print(len(data))
class NetflixDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for txt in txt_list:
            # Encode the descriptions using the GPT-Neo tokenizer
            encodings_dict = tokenizer(txt,
                                        truncation=True,
                                        max_length=max_length, 
                                        padding="max_length")
            input_ids = torch.tensor(encodings_dict['input_ids'])    
            self.input_ids.append(input_ids)
            mask = torch.tensor(encodings_dict['attention_mask'])
            self.attn_masks.append(mask)
    
    def __len__(self):
     return len(self.input_ids)

    def __getitem__(self, idx):
     return self.input_ids[idx], self.attn_masks[idx]

dataset = NetflixDataset(data, tokenizer, 512)

train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset, 
                            [train_size, len(dataset) - train_size])

training_args = TrainingArguments(output_dir="/export/data2/tdebets/models/test",
                                  num_train_epochs=1,
                                  logging_steps=1000,
                                  save_steps=1000,                                   
                                  per_device_train_batch_size=2,
                                  per_device_eval_batch_size=2,
                                  warmup_steps=100,
                                  weight_decay=0.01,  
                                  logging_dir="/export/data2/tdebets/logs/")

trainer = Trainer(model=model, args=training_args,  
                  train_dataset=train_dataset,
                  eval_dataset=val_dataset, 
                  # This custom collate function is necessary 
                  # to built batches of data
                  data_collator=lambda data: 
              {"input_ids": torch.stack([f[0] for f in data]),       
               "attention_mask": torch.stack([f[1] for f in data]),
               "labels": torch.stack([f[0] for f in data])})
# Start training process!
print('start training')
trainer.train()

tokenizer.save_pretrained('/export/data2/tdebets/tokenizer/gpt-neo_med')

generated = tokenizer.encode(
    f" <BOS> MY FATHER MEETS THE CAT  <newline>  <newline>  <newline>  One cold rainy day when my father was a little boy , he met an old  <newline>  alley cat on his street . <endprompt> <EOS>",
    return_tensors="pt").cuda()
model = model.cuda()

sample_outputs = model.generate(generated, do_sample=False, top_k=50, max_length=1024, top_p=0.95,
                                temperature=0, num_return_sequences=0, repetition_penalty=1.1)
# sample_outputs = model.generate(generated, max_length=50)
predicted_text2 = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
print(predicted_text2)