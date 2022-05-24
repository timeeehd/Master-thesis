import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('data/t5_data.csv', sep='\t', encoding='utf-8')[:51]

df2 = df.tail(50)
print(df.size)
from simplet5 import SimpleT5

model = SimpleT5()
model.from_pretrained(model_type="t5", model_name="t5-base")
# last_epoch_model = 't5' # put the name here
# model.load_model("t5", last_epoch_model, use_gpu=True)
# model.load_model("t5", last_epoch_model, use_gpu=True)
print('started training')
model.train(train_df=df,
            eval_df=df2, 
            source_max_token_len=256,
            target_max_token_len=1024, 
            batch_size=1, 
            max_epochs=1, 
            use_gpu=True
)