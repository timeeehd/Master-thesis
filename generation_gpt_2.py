from pathlib import Path
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
import transformers
transformers.logging.set_verbosity_error()


path = '/export/data2/tdebets/models/'
tokenizer_path = '/export/data2/tdebets/tokenizer/'
# maps = ['gpt-small2/']
# maps = ['gpt-neo/']

# maps = ['test_fairy/']

# numbers = [1,2,3,4]
# numbers = [0,5,22,50,75,100,125, 150]
numbers = [0]
with open('data/test (5).wp_source', "r", encoding='utf-8-sig') as file:
    data = file.readlines()
data = data[:300]
lenghts = [350]
for length in lenghts:
    print(length)
    for n in numbers:
        print(n)
        if n == 0:
            torch.manual_seed(42)
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            # model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-125M').cuda()
            model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
            print(next(model.parameters()).is_cuda)
            output = []
            output_w_prompt = []
            for d in tqdm(data[:300]):
                sent = d
        #     for i in range(12):
        #         print(i)
                torch.cuda.empty_cache()

            # generated = tokenizer.encode(f"<BOS> MY FATHER MEETS THE CAT  <newline>  <newline>  <newline>  One cold rainy day when my father was a little boy , he met an old  <newline>  alley cat on his street . <endprompt> <EOS>", return_tensors="pt")
                generated = tokenizer.encode(f'{sent}', return_tensors="pt").cuda()
            #         print(generated)
                sample_outputs = model.generate(generated, do_sample=True, top_k=50, max_length=length, top_p=0.95,
                        num_return_sequences=10, repetition_penalty=1.5, temperature=1.9, pad_token_id=tokenizer.eos_token_id)
                # sample_outputs = model.generate(generated, max_length=50)
                predicted_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
                sent = predicted_text.replace('\n',' <newline> ')
                output_w_prompt.append(sent)
                output.append(sent[len(d):])

            # f = open("results/med_FairyDB_test_2.txt", "w", encoding="utf-8")
            f = open(path + "results_wp/standard-small/" + str(length) + ".txt", "w", encoding="utf-8")

            for out in output:
                f.write(out + '\n')
            f.close()
            f = open(path + "results_wp/standard-small/prompt_" + str(length) + ".txt", "w", encoding="utf-8")
            # f = open("results/med_prompt_FairyDB_test_2.txt", "w", encoding="utf-8")
            for out in output_w_prompt:
                f.write(out + '\n')
            f.close()
        else:
            for map in maps:
                torch.manual_seed(42)
                print('initalize tokenizer')
                print(tokenizer_path + map)
                tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path + map)
                if 'neo' in map:
                    print('neo')
                    model = GPTNeoForCausalLM.from_pretrained(f'{path}{map}{str(n)}epochs').cuda()
                else:
                    print('initalize model')
                    model = GPT2LMHeadModel.from_pretrained(f"{path}{map}{str(n)}epochs").cuda()

                output = []
                output_w_prompt = []
                for d in tqdm(data[:300]):
                    sent = d
            #     for i in range(12):
            #         print(i)
                    torch.cuda.empty_cache()
                    if map == 'test/':
                # generated = tokenizer.encode(f"<BOS> MY FATHER MEETS THE CAT  <newline>  <newline>  <newline>  One cold rainy day when my father was a little boy , he met an old  <newline>  alley cat on his street . <endprompt> <EOS>", return_tensors="pt")
                        generated = tokenizer.encode(f'<BOS> {sent}', return_tensors="pt").cuda()
                #         print(generated)
                    else:
                        generated = tokenizer.encode(f'<BOS> <fairy> {sent}', return_tensors="pt").cuda()
                    sample_outputs = model.generate(generated, do_sample=True, top_k=50, max_length=length, top_p=0.95,
                            num_return_sequences=10, repetition_penalty=1.5, temperature=1.9, pad_token_id=tokenizer.eos_token_id)
                    # sample_outputs = model.generate(generated, max_length=50)
                    predicted_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
                    sent = predicted_text.replace('\n',' <newline> ')
                    output_w_prompt.append(sent)
                    output.append(sent[len(d):])

                # f = open("results/med_FairyDB_test_2.txt", "w", encoding="utf-8")
                f = open(path + "results_wp/" + map + str(n) + '_epochs_' + str(length) + ".txt", "w", encoding="utf-8")

                for out in output:
                    f.write(out + '\n')
                f.close()
                f = open(path + "results_wp/" + map + str(n) + '_epochs_prompt_' + str(length) + ".txt", "w", encoding="utf-8")
                # f = open("results/med_prompt_FairyDB_test_2.txt", "w", encoding="utf-8")
                for out in output_w_prompt:
                    f.write(out + '\n')
                f.close()