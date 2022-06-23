import numpy

from bert import *
import generator
import numpy as np
import pandas as pd

if __name__ == '__main__':
    gen = generator.init()
    discriminator = initialize_bert()
    # Load initial discriminator data
    data = pd.read_csv('../data/small_bert.csv', delimiter='\t', encoding='utf-8')
    df = data.copy(deep=True)
    df['story'].replace(to_replace='<newline>', value='', inplace=True, regex=True)
    print(f"number of newline tokens {df['story'].str.count('newline').sum()}")
    X_train, X_test, y_train, y_test = train_test_split(df['story'],
                                                        df['generated'],
                                                        test_size=0.1,
                                                        random_state=88)
    #Initial Discriminator training
    accuracy = 0
    while accuracy < 0.9:
        discriminator = train(discriminator, X_train,y_train, 1)
        loss, accuracy = discriminator.evaluate(X_test, y_test)
        print(f'Loss: {loss}')
        print(f'Accuracy: {accuracy}')

    # Loading prompts for generating, both for generation training and discriminator training
    with open('../data/Fairy_tales_generated_prompts_test.txt') as f:
        training = f.readlines()
    with open('../data/Fairy_tales_generated_prompts_test.txt') as f:
        test = f.readlines()

    # adversarial  training
    GAN_TRAINING_EPOCHS = 1
    for k in range(GAN_TRAINING_EPOCHS):
        score = []
        output = []
        output_w_prompt = []
        # Create generator training set
        for story in training[:1000]:
            prompt = story.strip()
            # change 200 to best performing reranker?
            samples = generator.sample(gen, prompt, 200)
            classified_results = classify(discriminator, samples)
            min_value = 1
            idx = 100
            for i in range(10):
                if tf.keras.backend.get_value(classified_results[i][0]) < min_value:
                    min_value = tf.keras.backend.get_value(classified_results[i][0])
                    idx = i
            score.append(min_value)
            output_w_prompt.append(prompt)
            output.append(samples[idx])

        # print('test')
        # score = np.array(score)
        # print(numpy.average(score, axis=0))
        # f = open('../results/test.txt', 'w', encoding='utf-8')
        # for out in output:
        #     f.write(out + '\n')
        # f.close()
        # # f = open("results/med_prompt_FairyDB_test_2.txt", "w", encoding="utf-8")
        # f = open('../results/test2.txt', 'w', encoding='utf-8')
        # for out in output_w_prompt:
        #     f.write(out + '\n')
        # f.close()

        df = pd.DataFrame(columns=['source_text', 'target_text'])
        df['source_text'] = output_w_prompt
        df['target_text'] = output

        df2 = df.tail(2)
        print('test')
        # train t5 again
        gen.train(train_df=df,
                    eval_df=df2,
                    source_max_token_len=256,
                    target_max_token_len=1024,
                    batch_size=1,
                    max_epochs=5,
                    use_gpu=True,
                    outputdir="/export/data2/tdebets/models/t5-base-test/",
                    )


        # Create discriminator training set + new results
        # save results of this
        score = []
        output = []
        output_w_prompt = []
        for story in test:
            prompt = story.strip()
            # change 200 to best performing reranker?
            samples = generator.sample(gen, prompt, 200)
            classified_results = classify(discriminator, samples)
            min_value = 1
            idx = 100
            for i in range(10):
                if tf.keras.backend.get_value(classified_results[i][0]) < min_value:
                    min_value = tf.keras.backend.get_value(classified_results[i][0])
                    idx = i
            print(min_value)
            score.append(min_value)
            output_w_prompt.append(prompt)
            output.append(samples[idx])
        # Save new dat
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
        #

        df = pd.DataFrame(columns=['story', 'generated'])
        df['story'] = output
        df['generated'] = 1
        # Still add gold text for training

        # change to correct data
        X_train, X_test, y_train, y_test = train_test_split(df['story'],
                                                            df['generated'],
                                                            test_size=0.1,
                                                            random_state=88)
        accuracy = 0
        while accuracy < 0.9:
            discriminator = train(discriminator, X_train, y_train, 1)
            loss, accuracy = discriminator.evaluate(X_test, y_test)
            print(f'Loss: {loss}')
            print(f'Accuracy: {accuracy}')
