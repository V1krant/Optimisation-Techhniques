import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

def random_data():
    x = np.random.uniform(0,1,100)
    y = x+0.1*np.random.randn(100)
    return (x, y)

x,y = random_data()

def costFunction(m, t0, t1, x, y):
	return 1/(2*m) * sum([(t0 + t1* np.asarray([x[i]]) - y[i])**2 for i in range(m)])


def sgd(learning_rate, x, y,iterations):
	theeta0 = 0
	theeta1 = 0

	m = x.shape[0]

	#total error
	J = costFunction(m, theeta0, theeta1, x, y)
	count = [i+1 for i in range(10000)]

	error = []
	theeta0=0
	theeat1=0

	for i in range(iterations):

		grad0 = 1/m * sum([(theeta0 + theeta1*np.asarray([x[i]]) - y[i]) for i in range(m)]) 
		grad1 = 1/m * sum([(theeta0 + theeta1*np.asarray([x[i]]) - y[i])*np.asarray([x[i]]) for i in range(m)])
		theeta0 = theeta0 - learning_rate * grad0
		theeta1 = theeta1 - learning_rate * grad1
		error.append(costFunction(m, theeta0, theeta1, x, y))

	return error, theeta0, theeta1

alpha = 0.01
num_iteration = 5000

error, theta0, theta1 = sgd(alpha, x, y,num_iteration)

#print(x,y)
print('Final  theta0 = ' + str(theta0)+"    ",end="")
print('Final  theta1 = ' + str(theta1))

plt.figure(0)
plt.scatter(x, y, c = 'red')
plt.plot(x, theta0 + theta1 * x)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dataset')
plt.savefig('output.png')
plt.show()