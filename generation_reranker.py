from transformers import T5Tokenizer,T5ForConditionalGeneration

from ranker_utilities import *

df = pd.read_csv('data/test.csv', sep='\t', encoding='utf-8')
data = [x for x in df['source_text']][:1]

preset_model = T5ForConditionalGeneration.from_pretrained(
                "t5-small", return_dict=True
            ).cuda()
tokenizer = T5Tokenizer.from_pretrained('t5-small')
own_model = T5ForConditionalGeneration.from_pretrained(
                "t5-small", return_dict=True
            ).cuda()
output = []
output_w_prompt = []
device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
for d in tqdm(data):
    prompt_length = len(d)
    sent = d.strip()
    for i in range(10):
        d = sent

        first_idx = False
        encodings_dict = tokenizer([sent])
        sliced_inputs = [encodings_dict['input_ids'][0][-550:]]

        prompts_ids = torch.tensor(
            sliced_inputs, device=device, dtype=torch.long)
        first_idx = len(prompts_ids[0]) if first_idx else 0
        max_length = len(prompts_ids[0]) + 25
        sample_outputs = own_model.generate(
            prompts_ids,  # Long tensor of size (batch_size, max_prompt_length)
            do_sample=True,  # activate top-k, top-p sampling
            max_length=max_length + first_idx,
            min_length=first_idx + max_length // 2 if first_idx else 10,
            top_k=70,
            top_p=0.95,
            temperature=1.05,
            repetition_penalty=1.0,  # no penalty
            num_return_sequences=10,
            pad_token_id=tokenizer.pad_token_id,
        )  # returns tensor of shape (len(prompts)*num_return_sequences x max_length)
        # first_idx = 0
        predicted_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
        has_space = d[0][-1].isspace()
        generated = map(lambda sample: preprocess_generated_text(
            sample[first_idx:], tokenizer, has_space), sample_outputs)
        generated = np.array(
            list(filter(lambda sample: len(sample.strip()) > 2, generated)))

        stories_scores = np.array(list(map(lambda text: score_text(
            text, tokenizer, preset_model, own_model), generated)))
        sorted_idx = sort_scores(stories_scores)

        sent = d + " " + list(generated[sorted_idx])[0].replace('\n', ' <newline> ')
    print(len(sent))
    print(prompt_length)
    output_w_prompt.append(sent)
    output.append(sent[prompt_length:])
    # f = open("results/med_FairyDB_test_2.txt", "w", encoding="utf-8")
# f = open(path + "results_wp/standard-t5-med/" + str(length) + ".txt", "w", encoding="utf-8")
f = open('results/test.txt', 'w', encoding='utf-8')
for out in output:
    f.write(out + '\n')
f.close()
# f = open("results/med_prompt_FairyDB_test_2.txt", "w", encoding="utf-8")
f = open('results/test2.txt', 'w', encoding='utf-8')
for out in output_w_prompt:
    f.write(out + '\n')
f.close()

