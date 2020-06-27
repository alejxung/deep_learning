import pandas as pd
import os
import numpy as np
import random
from sklearn import preprocessing
from collections import deque
import time
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

SEQ_LENGTH = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = "ETH-USD" # ["BTC-USD", "LTC-USD", "ETH-USD", "BCH-USD"]
BATCH_SIZE = 64
EPOCHS = 10
NAME = "{}-{}-SEQ-{}-PRED-{}".format(RATIO_TO_PREDICT, SEQ_LENGTH, FUTURE_PERIOD_PREDICT, int(time.time()))

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

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_validation = np.asarray(x_validation)
y_validation = np.asarray(y_validation)



print("train data: {}, validation: {}".format(len(x_train), len(x_validation)))
print("Dont buys: {}, buys: {}".format(np.ndarray.tolist(y_train).count(0), np.ndarray.tolist(y_train).count(1)))
print("VALIDATION Dont buys: {}, buys: {}".format(np.ndarray.tolist(y_validation).count(0), np.ndarray.tolist(y_validation).count(1)))


model = Sequential()
model.add(LSTM(128, activation="relu", input_shape=(x_train.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, activation="relu", input_shape=(x_train.shape[1:]), return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128, activation="relu", input_shape=(x_train.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(2, activation="softmax"))

optimizer = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
file_path = "RNN_Final-{epoch:02d}-{val_acc:.3f}" # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("models/{}.model".format(file_path, monitor="val_acc", verbose=1, save_best_only=True, mode="max")) # saves only the best ones

history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_validation, y_validation), callbacks=[tensorboard, checkpoint])