import torch
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPTNeoForCausalLM
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F
import csv
import os
import sys


class Stories(Dataset):
    def __init__(self, df, control_code='', truncate=False, gpt2_type="gpt2", max_length=2048):

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        #         self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.stories = []

        for row in df:
            self.stories.append(torch.tensor(
                self.tokenizer.encode(row[:max_length])
            ))
        if truncate:
            self.stories = self.stories[:20000]
        self.story_count = len(self.stories)

    def __len__(self):
        return self.story_count

    def __getitem__(self, item):
        return self.stories[item]

with open('data/Fairy_tales_combined (1).txt', "r", encoding='utf-8-sig') as file:
    data = file.readlines()


dataset = Stories(data, truncate=True, gpt2_type="gpt2")


#Get the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
# model.resize_token_embeddings(len(tokenizer))
# model.load_state_dict(torch.load('models/GPT2-med-2048-512.pt', map_location=torch.device('cpu')))
# model.load_state_dict(torch.load('models/GPT2-small.pt'))


special_tokens_dict = {
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "pad_token": "<PAD>",
        "additional_special_tokens": [
            "<endprompt>",
        ],
    }

num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None


def train(
        dataset, model, tokenizer,
        batch_size=4, epochs=100, lr=2e-5,
        max_seq_len=2048, warmup_steps=50,
        gpt2_type="gpt2", output_dir=".", output_prefix="wreckgar",
        test_mode=False, save_model_on_epoch=False,
):
    acc_steps = 100
    device = torch.device("cuda")
    model = model.cuda()
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )
    torch.cuda.empty_cache()
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    loss = 0
    accumulating_batch_count = 0
    input_tensor = None
    torch.cuda.empty_cache()
    for epoch in range(epochs):
        torch.cuda.empty_cache()
        sys.stdout.write(f"Training epoch {epoch}")
        print(loss)
        for idx, entry in (enumerate(train_dataloader)):
            (input_tensor, carry_on, remainder) = pack_tensor(entry, input_tensor, 1024)

            if carry_on and idx != len(train_dataloader) - 1:
                continue

            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor, labels=input_tensor)
            print(outputs)
            loss = outputs[0]
            print(loss)
            print(loss.size)
            loss.backward()

            if (accumulating_batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            accumulating_batch_count += 1
            input_tensor = None
        if save_model_on_epoch:
            torch.save(
                model.state_dict(),
                #                 os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
                os.path.join('/export/data2/tdebets/models', f"{output_prefix}.pt"),
            )
            tokenizer.save_pretrained('models/tokenizer/gpt2/')

    return model
sys.stdout.write('start training')
model = train(dataset, model, tokenizer, save_model_on_epoch = True, output_prefix= 'GPT2-small-2048-1024')
sys.stdout.write('finished training')