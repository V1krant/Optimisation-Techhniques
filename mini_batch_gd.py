import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

# dataset = pd.read_csv("dataset.csv", index_col = 0)
# x = np.array(dataset['x'])
# y = np.array(dataset['y'])
def random_data():
    x = np.random.uniform(0,1,100)
    y = 100*(x+0.1*np.random.randn(100))
    return (x, y)

x,y = random_data()

def costFunction(m, t0, t1, x, y):
	return 1/(2*m) * sum([(t0 + t1* np.asarray([x[i]]) - y[i])**2 for i in range(m)])

def gen_mini_batches(x, y, batch_size): 
	mini_batches = [] 
	data = np.hstack((x, y)) 
	np.random.shuffle(data) 
	n_minibatches = data.shape[0]
	i = 0
  
	for i in range(n_minibatches + 1): 
		mini_batch = data[i * batch_size:(i + 1)*batch_size, :] 
		x_mini = mini_batch[:, :-1] 
		Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
		mini_batches.append((x_mini, Y_mini)) 
	if data.shape[0] % batch_size != 0: 
		mini_batch = data[i * batch_size:data.shape[0]] 
		x_mini = mini_batch[:, :-1] 
		Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
		mini_batches.append((x_mini, Y_mini)) 
	return mini_batches 


def mini_batch_gd(learning_rate, x, y, batch_size,iterations):
	theeta0 = 0
	theeta1 = 0

	m = x.shape[0]

	#total error
	J = costFunction(m, theeta0, theeta1, x, y)
	count = [i+1 for i in range(10000)]

	error = []

	for i in range(iterations):
		mini_batches = gen_mini_batches(x, y, batch_size) 
		for mini_batch in mini_batches: 
			grad0 = 1/m * sum([(theeta0 + theeta1*np.asarray([x[i]]) - y[i]) for i in range(m)]) 
			grad1 = 1/m * sum([(theeta0 + theeta1*np.asarray([x[i]]) - y[i])*np.asarray([x[i]]) for i in range(m)])

			#Updating the parameters
			theeta0 = theeta0 - learning_rate * grad0
			theeta1 = theeta1 - learning_rate * grad1
			error.append(cost(m, theeta0, theeta1, x, y))

	return error, theeta0, theeta1

alpha = 0.01
num_iteration = 10000
batch_size=40

error, theta0, theta1 = mini_batch_gd(alpha, x, y,batch_size,num_iteration)

print('theta0 = ' + str(theta0))
print('theta1 = ' + str(theta1))

plt.figure(0)
plt.scatter(x, y, c = 'red')
plt.plot(x, theta0 + theta1 * x)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dataset')
plt.savefig('output.png')
plt.show()

plt.figure(1)
plt.plot(count, loss)
plt.xlabel('iteration')
plt.ylabel('Loss')
plt.title('Loss History')
plt.savefig('loss.png')
plt.show()
