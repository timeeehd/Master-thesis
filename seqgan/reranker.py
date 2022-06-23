import numpy

from bert import *
import generator
import numpy as np

if __name__ == '__main__':
    gen = generator.init()
    discriminator = initialize_bert()
    data = pd.read_csv('../data/small_bert.csv', delimiter='\t', encoding='utf-8')
    df = data.copy(deep=True)
    df['story'].replace(to_replace='<newline>', value='', inplace=True, regex=True)
    print(f"number of newline tokens {df['story'].str.count('newline').sum()}")
    X_train, X_test, y_train, y_test = train_test_split(df['story'],
                                                        df['generated'],
                                                        test_size=0.1,
                                                        random_state=88)
    accuracy = 0
    while accuracy < 0.9:
        discriminator = train(discriminator, X_train,y_train, 1)
        loss, accuracy = discriminator.evaluate(X_test, y_test)
        print(f'Loss: {loss}')
        print(f'Accuracy: {accuracy}')

    with open('../data/Fairy_tales_generated_prompts_test.txt') as f:
        stories = f.readlines()
    score = []
    output = []
    output_w_prompt = []
    for story in stories[:2]:
        prompt = story.strip()
        for j in range(10):
            samples = generator.sample(gen, prompt, 25)
            classified_results = classify(discriminator, samples)
            min_value = 1
            idx = 100
            for i in range(10):
                if tf.keras.backend.get_value(classified_results[i][0]) < min_value:
                    min_value = tf.keras.backend.get_value(classified_results[i][0])
                    idx = i
            prompt = prompt + ' ' + samples[idx]
        print(min_value)
        score.append(min_value)
        output_w_prompt.append(prompt)
        output.append(prompt[len(story.strip()):])
    print('test')
    score = np.array(score)
    print(numpy.average(score, axis=0))
    f = open('../results/test.txt', 'w', encoding='utf-8')
    for out in output:
        f.write(out + '\n')
    f.close()
    # f = open("results/med_prompt_FairyDB_test_2.txt", "w", encoding="utf-8")
    f = open('../results/test2.txt', 'w', encoding='utf-8')
    for out in output_w_prompt:
        f.write(out + '\n')
    f.close()
