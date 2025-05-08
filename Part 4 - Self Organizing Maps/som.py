import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show
from matplotlib.patches import Rectangle
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Importing the dataset
dataset = pd.read_csv('Part 4 - Self Organizing Maps/dataset/Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values # everything except last column
y = dataset.iloc[:, -1].values # all rows but only the last column

# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1)) # scaling the values of X between 0 and 1
X = sc.fit_transform(X)

# Training the SOM
som = MiniSom(x = 12, y = 12, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualising the Results
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, 
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
    
threshold = 0.8
distance_map = som.distance_map()
high_distance_nodes = np.argwhere(distance_map > threshold)

# Get the current axes
ax = plt.gca()

# Add a square border for each high-distance node
for node in high_distance_nodes:
    rect = Rectangle((node[0], node[1]), 1, 1,
                     linewidth=2,
                     edgecolor='orange',
                     facecolor='none')
    ax.add_patch(rect)
    
show()

# Finding the Frauds
mappings = som.win_map(X)
frauds = np.concatenate([mappings[tuple(node)] for node in high_distance_nodes if tuple(node) in mappings], axis=0)
frauds = sc.inverse_transform(frauds)
print("Suspected Fraudlent Customer IDs:")
for x, i in enumerate(frauds):
    is_last = (x + 1 == len(frauds))
    is_line_end = (x + 1) % 5 == 0
    end_char = '' if is_last else (',\n' if is_line_end else ', ')
    print('{0:.0f}'.format(frauds[x, 0]), end=end_char)

customers = dataset.iloc[:, 1:].values

is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1

sc = StandardScaler()
customers = sc.fit_transform(customers)

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=2, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
ann.fit(customers, is_fraud, batch_size = 1, epochs = 10)

y_pred = ann.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1)
y_pred = y_pred[y_pred[:, 1].argsort()]

print(y_pred)