# based on "Python Machine Learning by Sebastian Raschka, 2015".

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

directory = "C:/Users/me/Documents/psi/bogobogo/"
sys.path.append(directory)

from Perceptron import Perceptron


# wektor danych destujacych tzn testujemy czy literka duza 'A' zostanie rozpoznana
A = [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1];
    
a=[0, 0, 0, 0, 0, 0,0,0,0,0, 0,0,0,0,0, 1,1,1,1,0, 1,0,0,1,0, 1,0,0,1,0, 1,1,1,1,1 ];
b=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0,0,1, 1,0,0,0,1, 1,1,1,1,0 ];
c=[0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 1,1,1,1,0, 1,0,0,0,0,1,0,0,0,0, 1,1,1,1,0];
d=[0,0,0,0,1, 0,0,0,0,1, 0,0,0,0,1, 0,1,1,1,1, 1,0,0,0,1, 1,0,0,0,1, 0,1,1,1,1 ];
e=[0,0,0,0,0, 0,0,0,0,0 ,1,1,1,1,0, 1,0,0,0,1, 1,1,1,1,1, 1,0,0,0,0, 0,1,1,1,1 ];
f=[0,0,0,0,0, 0,1,1,1,0, 0,1,0,0,0, 1,1,1,0,0, 0,1,0,0,0, 0,1,0,0,0, 0,1,0,0,0 ];
g=[0,1,1,1,0, 0,1,0,1,0, 0,1,1,1,1, 0,0,0,1,0, 0,0,1,1,0, 0,1,0,1,0,0,1,1,1,0 ];
h=[1,0,0,0,0, 1,0,0,0,0, 1,0,0,0,0, 1,1,1,1,0, 1,0,0,1,0, 1,0,0,1,0, 1,0,0,1,0 ];
i=[0,0,0,0,0, 0,0,1,0,0, 0,0,0,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0 ];

# macierz danych uczacych duzych i malych liter
input = np.array([A, a, b, c, d, e, f, g, h, i])

# wektor danych wyjsciowych, takich jakie chcemy uzyskac na wyjscie neuroru. 1 dla duzej litery, 0 dla malej
output =  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];   # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

errors = []
eta = 0.2
n = 10

pn = Perceptron(eta, n)
pn.fit(input, output)
plt.plot(range(1, len(pn.errors) + 1), pn.errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()

# for big letters 1, for smaller letters -1
print pn.predict(input[2])
