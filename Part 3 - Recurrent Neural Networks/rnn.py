import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import keras._tf_keras.keras as keras
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, LSTM, Dropout

dataset_train = pd.read_csv('Part 3 - Recurrent Neural Networks/dataset/Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

rnn = Sequential([
    keras.Input(shape=(X_train.shape[1], 1)),
    keras.layers.LSTM(units = 65, return_sequences = True),
    keras.layers.Dropout(0.18),
    keras.layers.LSTM(units = 65, return_sequences = True),
    keras.layers.Dropout(0.18),
    keras.layers.LSTM(units = 65, return_sequences = True),
    keras.layers.Dropout(0.18),
    keras.layers.LSTM(units = 65),
    keras.layers.Dropout(0.18),
    keras.layers.Dense(units = 1),
])

rnn.compile(optimizer = 'adam', loss = 'mean_squared_error')

rnn.fit(X_train, y_train, epochs = 100, batch_size = 16)

dataset_test = pd.read_csv('Part 3 - Recurrent Neural Networks/dataset/Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = rnn.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()