import numpy

from bert import *
import generator
import numpy as np

if __name__ == '__main__':
    tokenizer, gen = generator.init()
    discriminator = initialize_bert()
    data = pd.read_csv('../data/small_bert2.csv', delimiter='\t', encoding='utf-8')
    df = data.copy(deep=True)
    df['story'].replace(to_replace='<newline>', value='', inplace=True, regex=True)
    print(f"number of newline tokens {df['story'].str.count('newline').sum()}")
    X_train, X_test, y_train, y_test = train_test_split(df['story'],
                                                        df['human'],
                                                        test_size=0.1,
                                                        random_state=88)
    epochs = 2
    discriminator = train(discriminator, X_train,y_train, epochs)
    loss, accuracy = discriminator.evaluate(X_test, y_test)
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')
    with open('../data/Fairy_tales_generated_prompts_test.txt') as f:
        stories = f.readlines()
    score = []
    for story in stories[:100]:
        neg_examp, samples = generator.sample(gen,tokenizer, [story])
        classified_results = classify(discriminator, [neg_examp[0]])
        score.append(tf.keras.backend.get_value(classified_results[0][0]))
    print(score)
    score = np.array(score)
    print(numpy.average(score, axis=0))

    # classified_results = classify(discriminator, [pos_examp, neg_examp[0]])
    # print(classified_results)
    gen = generator.train(gen, discriminator, tokenizer, data=stories)
    score = []
    for story in stories[:100]:
        neg_examp, samples = generator.sample(gen,tokenizer, [story])
        classified_results = classify(discriminator, [neg_examp[0]])
        score.append(tf.keras.backend.get_value(classified_results[0][0]))
    print(score)
    score = np.array(score)
    print(numpy.average(score, axis=0))
    gen = generator.train(gen, discriminator, tokenizer, data=stories, epochs=40)
    score = []
    for story in stories[:100]:
        neg_examp, samples = generator.sample(gen,tokenizer, [story])
        classified_results = classify(discriminator, [neg_examp[0]])
        score.append(tf.keras.backend.get_value(classified_results[0][0]))
    print(score)
    score = np.array(score)
    print(numpy.average(score, axis=0))
    # neg_examp, samples = generator.sample(gen,tokenizer)
    # classified_results = classify(discriminator, [pos_examp, neg_examp[0]])
    # print(classified_results)
    # tf.keras.backend.get_value(classified_results[1][0])
    sample_outputs = model.generate(
        prompts_ids,  # Long tensor of size (batch_size, max_prompt_length)
        do_sample=True,  # activate top-k, top-p sampling
        max_length=max_length + first_idx,
        min_length=first_idx + max_length // 2 if first_idx else 10,
        top_k=70,
        top_p=0.95,
        temperature=1.05,
        repetition_penalty=1.0,  # no penalty
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
    )