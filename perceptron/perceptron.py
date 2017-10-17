#import matplotlib.pyplot as plt
#from pylab import plot, ylim, show
from random import choice
from numpy import array, dot, random

activ_func = lambda x: 0 if x < 0 else 1 #funkcja jednostkowa(aktywacji)

training_data = [
	(array([0,0,1]), 0),
	(array([0,1,1]), 1),
	(array([1,0,1]), 1),
	(array([1,1,1]), 1),

]
#wybieramy trzy losowe liczby w przedziale 0,1
w = random.rand(3)  # jako wagi poczatkowe

errors = []
eta = 0.2
n = 100

for i in xrange(n):
	x, expected = choice(training_data)
	result = dot(w, x)
	error = expected - activ_func(result)
	w += eta * error * x
	
for x, _ in training_data:
	result = dot(x, w)
	print "{}: {} -> {}".format(x[:2], result, activ_func(result))
	

#n_point = arange(n)

#plt.plot(errors, n)
#ylim([-1,1])
#plot(errors)
#show()	
