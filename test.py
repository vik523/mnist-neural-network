import matplotlib.pyplot as plt
import gzip
import pickle
from neural_net import Network
import numpy as np

ft = gzip.open('data_testing', 'rb')
TESTING = pickle.load(ft)
ft.close()

nt1=Network(300)
nt1.load('model.net')
plt.imshow(TESTING[0][0].reshape((28,28)), cmap='gray')
plt.text(1,1, 'Result is ' + str(np.argmax(nt1.predict(TESTING[0][0]))), color='white')
plt.show()
plt.pause(10)
print('Result is ', np.argmax(nt1.predict(TESTING[0][0])))
print('finish')