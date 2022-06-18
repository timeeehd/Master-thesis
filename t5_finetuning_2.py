import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('data/t5_data_2.csv', sep='\t', encoding='utf-8')

df2 = df.tail(50)
print(df.size)
from simple_t5_own import SimpleT5_own

model = SimpleT5_own()
print('created own t5')
model.from_pretrained(model_type="t5", model_name="t5-base", new_token=True)
last_epoch_model = '/export/data2/tdebets/models/t5/100epochs/' # put the name here
# model.load_model("t5", last_epoch_model, use_gpu=True)
# model.load_model("t5", last_epoch_model, use_gpu=True)
print('started training')
model.train(train_df=df,
            eval_df=df2, 
            source_max_token_len=256,
            target_max_token_len=1024,
            batch_size=1, 
            max_epochs=50, 
            use_gpu=True,
            outputdir = "/export/data2/tdebets/models/t5-base-test/",
)