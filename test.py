import matplotlib.pyplot as plt
import gzip
import pickle
from neural_net import Network
import numpy as np
import random

ft = gzip.open('data_testing', 'rb')
TESTING = pickle.load(ft)
ft.close()
index = random.randint(1, len(TESTING[0]))
nt1=Network(300)
nt1.load('model.net')
plt.imshow(TESTING[0][index].reshape((28,28)), cmap='gray')
plt.text(1,1, 'Result is ' + str(np.argmax(nt1.predict(TESTING[0][index]))) + ', real value is ' + str(TESTING[1][index]), color='white')
plt.show()
plt.pause(1)
print('Result is ', np.argmax(nt1.predict(TESTING[0][index])))
print('finish')