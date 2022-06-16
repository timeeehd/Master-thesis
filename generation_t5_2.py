import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from simplet5 import SimpleT5
from tqdm import tqdm

with open('data/test (5).wp_source', "r", encoding='utf-8-sig') as file:
    data = file.readlines()
data = data[:300]
path = '/export/data2/tdebets/models/'
maps = ['t5/']
numbers = [0, 5, 22, 75, 100, 125, 150]
lenghts = [200,350]

for length in lenghts:
    print(length)
    for n in numbers:
        print(n)
        if n == 0:
            model = SimpleT5()
            model.from_pretrained(model_type="t5", model_name="t5-small")
            model.load_model("t5", 't5-small', use_gpu=False)
            output = []
            output_w_prompt = []
            for d in tqdm(data):
                sent = d
                predicted_text = model.predict(sent, max_length=length, repetition_penalty =1.5,
                top_k=50, top_p=0.95)[0]
                sent = predicted_text.replace('\n',' <newline> ')
                output_w_prompt.append(d + " "+ sent)
                output.append(sent)
                        # f = open("results/med_FairyDB_test_2.txt", "w", encoding="utf-8")
            f = open(path + "results_wp/standard-t5/" + str(length) + ".txt", "w", encoding="utf-8")

            for out in output:
                f.write(out + '\n')
            f.close()
            f = open(path + "results_wp/standard-t5/prompt_" + str(length) + ".txt", "w", encoding="utf-8")
            # f = open("results/med_prompt_FairyDB_test_2.txt", "w", encoding="utf-8")
            for out in output_w_prompt:
                f.write(out + '\n')
            f.close()
        else:
            for map in maps:
                model = SimpleT5()
                model.from_pretrained(model_type="t5", model_name="t5-small")
                last_epoch_model = f'{path}{map}{str(n)}epochs' # put the name here
                # model.load_model("t5", last_epoch_model, use_gpu=True)
                model.load_model("t5", last_epoch_model, use_gpu=False)
                print('started training')
                output = []
                output_w_prompt = []
                for d in tqdm(data):
                    sent = d
                    predicted_text = model.predict(sent, max_length=length, repetition_penalty =1.5,
                    top_k=50,  top_p=0.95)[0]  
                    sent = predicted_text.replace('\n',' <newline> ')
                    output_w_prompt.append(d + " "+ sent)
                    output.append(sent)
                            # f = open("results/med_FairyDB_test_2.txt", "w", encoding="utf-8")
                f = open(path + "results_wp/" + map + str(n)+ '_epochs_'+str(length) + ".txt", "w", encoding="utf-8")

                for out in output:
                    f.write(out + '\n')
                f.close()
                f = open(path + "results_wp/" + map + str(n)+ '_epochs_prompt_'+str(length) + ".txt", "w", encoding="utf-8")
                # f = open("results/med_prompt_FairyDB_test_2.txt", "w", encoding="utf-8")
                for out in output_w_prompt:
                    f.write(out + '\n')
                f.close()
