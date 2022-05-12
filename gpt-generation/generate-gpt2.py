from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPTNeoForCausalLM
import torch

with open('data/Fairy_tales_generated_prompts_test.txt', "r", encoding='utf-8-sig') as file:
    data = file.readlines()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

# special_tokens_dict = {
#         "bos_token": "<BOS>",
#         "eos_token": "<EOS>",
#         "pad_token": "<PAD>",
#         "additional_special_tokens": [
#             "<endprompt>",
#         ],
#     }
#
# num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
# model.resize_token_embeddings(len(tokenizer))
# torch.manual_seed(42)

def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None

output = []
# model = GPT2LMHeadModel.from_pretrained('models/GPT2-small-1024.pt')
# model = torch.load('models/GPT2-small-1024.pt', map_location=torch.device('cpu'))
# model.load_state_dict(torch.load('models/GPT2-small-2048-1024.pt', map_location=torch.device('cpu')))
# model.eval()
for d in data[:1]:
    # generated = tokenizer.encode(f"<BOS> MY FATHER MEETS THE CAT  <newline>  <newline>  <newline>  One cold rainy day when my father was a little boy , he met an old  <newline>  alley cat on his street . <endprompt> <EOS>", return_tensors="pt")
    generated = tokenizer.encode('Hello my name is Tim!', return_tensors="pt")
    sample_outputs = model.generate(generated, do_sample=False, top_k=50, max_length=1024, top_p=0.95,
            num_return_sequences=0, repetition_penalty=1.5)
    # sample_outputs = model.generate(generated, max_length=50)
    predicted_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
    output.append(predicted_text)

for out in output:
    print(out + '\n')


from transformers import pipeline
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-125M')
print(generator("Hello my name is Tim!", do_sample=True, min_length=50))


