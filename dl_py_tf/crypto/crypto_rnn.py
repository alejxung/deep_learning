#%%

import pandas as pd
import os
import numpy as np
import random
from sklearn import preprocessing
from collections import deque

SEQ_LENGTH = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = "LTC-USD"

def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

def preprocess_df(df):   
    df = df.drop("future", 1)
    
    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LENGTH)

    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LENGTH:
            sequential_data.append([np.array(prev_days), i[-1]])
    
    random.shuffle(sequential_data)

    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        if target == 1:
            buys.append([seq, target])
    
    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))

    # if len in 30k
    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys+sells
    random.shuffle(sequential_data)

    x = []
    y = []

    for seq, target in sequential_data:
        x.append(seq)
        y.append(target)
        
    return np.array(x), y



main_df = pd.DataFrame()

ratios = ["BTC-USD", "LTC-USD", "ETH-USD", "BCH-USD"]
for ratio in ratios:
    dataset = "crypto_data/{}.csv".format(ratio)
   
    df = pd.read_csv(dataset, names=["time", "low", "high", "open", "close", "volume"])
    df.rename(columns={"close": "{}_close".format(ratio), "volume": "{}_volume".format(ratio)}, inplace=True)
   
    df.set_index("time", inplace=True)
    df = df[["{}_close".format(ratio), "{}_volume".format(ratio)]]

    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

main_df["future"] = main_df["{}_close".format(RATIO_TO_PREDICT)].shift(-FUTURE_PERIOD_PREDICT)
main_df["target"] = list(map(classify, main_df["{}_close".format(RATIO_TO_PREDICT)], main_df["future"]))
# print(main_df[["{}_close".format(RATIO_TO_PREDICT), "future", "target"]].head(10))

times = sorted(main_df.index.values)
last_5pct = times[-int(0.05*len(times))]
# print(last_5pct)

validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

x_train, y_train = preprocess_df(main_df)
x_validation, y_validation = preprocess_df(validation_main_df)

print("train data: {}, validation: {}".format(len(x_train), len(x_validation)))
print("Dont buys: {}, buys: {}".format(y_train.count(0), y_train.count(1)))
print("VALIDATION Dont buys: {}, buys: {}".format(y_validation.count(0), y_validation.count(1)))