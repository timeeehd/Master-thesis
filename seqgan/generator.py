from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re
import torch
import numpy as np
import tensorflow as tf
from bert import *
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import sys
from simple_t5_own import SimpleT5_own


def _preprocess_generated_text(sample, tokenizer, has_space):
    decoded = tokenizer.decode(
        sample, skip_special_tokens=True)
    # Removing spaces.
    decoded = decoded.strip()
    # Adding a space at the beginning if needed.
    if not has_space:
        decoded = ' ' + decoded
    # Filtering � globally
    return re.sub(u'\uFFFD', '', decoded)


def init():
    model = SimpleT5_own()
    print('created own t5')
    model.from_pretrained(model_type="t5", model_name="t5-base", new_token=True)
    # last_epoch_model = '/export/data2/tdebets/models/t5/100epochs/'  # put the name here
    model.load_model("t5", 't5-base', use_gpu=True)
    return model


def sample(model, sentence, max_length):
    predicted_text = model.predict(sentence,  # Long tensor of size (batch_size, max_prompt_length)
                                   do_sample=True,  # activate top-k, top-p sampling
                                   max_length=max_length,
                                   top_k=70,
                                   top_p=0.95,
                                   repetition_penalty=1.0,  # no penalty
                                   num_return_sequences=10,
                                   num_beams=25,
                                   )
    return predicted_text

def calculate_loss(results, tensors):
    loss = 0
    for j in range(len(tensors)):
        for i in range(len(tensors[j])):
            res = tf.keras.backend.get_value(results[j][0])
            loss += res
    return loss / len(tensors)


def train(gen, discr, tokenizer, epochs=10, lr=2e-5,
          max_seq_len=2048, warmup_steps=50, data=None):
    gen.train()
    optimizer = AdamW(gen.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )
    for epoch in range(epochs):
        sys.stdout.write(f"Training epoch {epoch}")
        texts, tensors = sample(gen, tokenizer, [data[126]])
        # print(texts)
        classified_results = classify(discr, texts)
        # print(classified_results)
        outputs = gen(tensors, labels=tensors)
        calculated_loss = calculate_loss(classified_results, tensors)
        loss = torch.tensor([-calculated_loss], requires_grad=True)
        print(loss)
        loss.backward()
        optimizer.zero_grad()
        gen.zero_grad()
    return gen
