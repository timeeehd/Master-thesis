import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPTNeoForCausalLM
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F
import csv
import os


class Stories(Dataset):
    def __init__(self, df, control_code='', truncate=False, gpt2_type="gpt2-medium", max_length=2048):

        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
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


# class Stories2(Dataset):
#     def __init__(self, df, control_code= '', truncate=False, gpt2_type="EleutherAI/gpt-neo-125M", max_length=1048):

#         self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
#         self.stories = []

#         for row in df:
#           self.stories.append(torch.tensor(
#                 self.tokenizer.encode(data[:max_length])
#             ))
#         if truncate:
#             self.stories = self.stories[:20000]
#         self.story_count = len(self.stories)

#     def __len__(self):
#         return self.story_count

#     def __getitem__(self, item):
#         return self.stories[item]

with open('data/Fairy_tales_combined (1).txt', "r", encoding='utf-8-sig') as file:
    data = file.readlines()
# import pandas as pd

# data2 = pd.read_csv('../input/lm-finetuning/Fairy_tales_combined (1).txt', sep = '\n', header=None)
# data2.head()

dataset = Stories(data[:500], truncate=True, gpt2_type="gpt2")
# dataset2 = Stories2(data, truncate=True, gpt2_type="EleutherAI/gpt-neo-125M")

# print(dataset[2])
# dataset2[2]


#Get the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# special_tokens_dict = {
#         "bos_token": "<BOS>",
#         "eos_token": "<EOS>",
#         # "pad_token": "<PAD>",
#         # "additional_special_tokens": [
#         #     "<endprompt>",
#         # ],
#     }

# num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
# model.resize_token_embeddings(len(tokenizer))

#Accumulated batch size (since GPT2 is so big)
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
    batch_size=16, epochs=1, lr=2e-5,
    max_seq_len=2048, warmup_steps=50,
    gpt2_type="gpt2", output_dir=".", output_prefix="wreckgar",
    test_mode=False,save_model_on_epoch=False,
):
    acc_steps = 100
    device=torch.device("cpu")
    # model = model.cuda()
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )
    torch.cuda.empty_cache()
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    loss=0
    accumulating_batch_count = 0
    input_tensor = None
    torch.cuda.empty_cache()
    for epoch in range(epochs):
        torch.cuda.empty_cache()
        print(f"Training epoch {epoch}")
        print(loss)
        for idx, entry in tqdm(enumerate(train_dataloader)):
            (input_tensor, carry_on, remainder) = pack_tensor(entry, input_tensor, 512)

            if carry_on and idx != len(train_dataloader) - 1:
                continue

            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor, labels=input_tensor)
            loss = outputs[0]
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
                os.path.join(output_dir, f"{output_prefix}.pt"),
            )
    return model

model = train(dataset, model, tokenizer, save_model_on_epoch = True, output_prefix= 'test')

generated = tokenizer.encode(
    f" MY FATHER MEETS THE CAT  <newline>  <newline>  <newline>  One cold rainy day when my father was a little boy , he met an old  <newline>  alley cat on his street . ",
    return_tensors="pt")
sample_outputs = model.generate(generated, do_sample=False, top_k=50, max_length=1024, top_p=0.95,
                                temperature=0, num_return_sequences=0, repetition_penalty=1.1)
# sample_outputs = model.generate(generated, max_length=50)
predicted_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
print(predicted_text)

generated = tokenizer.encode(
    f" <BOS> MY FATHER MEETS THE CAT  <newline>  <newline>  <newline>  One cold rainy day when my father was a little boy , he met an old  <newline>  alley cat on his street . <EOS>",
    return_tensors="pt")
sample_outputs = model.generate(generated, do_sample=False, top_k=50, max_length=1024, top_p=0.95,
                                temperature=0, num_return_sequences=0, repetition_penalty=1.1)
# sample_outputs = model.generate(generated, max_length=50)
predicted_text2 = tokenizer.decode(sample_outputs[0], skip_special_tokens=False)
print(predicted_text2)