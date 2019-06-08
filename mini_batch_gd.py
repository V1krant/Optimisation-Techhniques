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
		# print(data) 
		# print(data[i * batch_size:(i + 1)*batch_size :] )
		mini_batch = data[i * batch_size:(i + 1)*batch_size :] 
		x_mini = mini_batch[: :-1] 
		Y_mini = mini_batch[: -1].reshape((-1, 1)) 
		mini_batches.append((x_mini, Y_mini)) 
	if data.shape[0] % batch_size != 0: 
		mini_batch = data[i * batch_size:data.shape[0]] 
		x_mini = mini_batch[: :-1] 
		Y_mini = mini_batch[: -1].reshape((-1, 1)) 
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
			error.append(costFunction(m, theeta0, theeta1, x, y))

	return error, theeta0, theeta1

alpha = 0.01
num_iteration = 100
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
plt.plot(error, loss)
plt.xlabel('iteration')
plt.ylabel('Loss')
plt.title('Loss History')
plt.savefig('loss.png')
plt.show()

# linear regression using "mini-batch" gradient descent 
# function to compute hypothesis / predictions 
# linear regression using "mini-batch" gradient descent 
# function to compute hypothesis / predictions 
# import numpy as np 
# import matplotlib.pyplot as plt   
# # creating data 
# mean = np.array([5.0, 6.0]) 
# cov = np.array([[1.0, 0.95], [0.95, 1.2]]) 
# data = np.random.multivariate_normal(mean, cov, 8000) 
  
# # visualising data 
# #plt.scatter(data[:500, 0], data[:500, 1], marker = '.') 
# ##plt.show() 
  
# # train-test-split 
# data = np.hstack((np.ones((data.shape[0], 1)), data)) 
  
# split_factor = 0.90
# split = int(split_factor * data.shape[0]) 
  
# X_train = data[:split, :-1] 
# y_train = data[:split, -1].reshape((-1, 1)) 
# X_test = data[split:, :-1] 
# y_test = data[split:, -1].reshape((-1, 1)) 
  
# print("Number of examples in training set = % d"%(X_train.shape[0])) 
# print("Number of examples in testing set = % d"%(X_test.shape[0]))
# def hypothesis(X, theta): 
# 	return np.dot(X, theta) 

# # function to compute gradient of error function w.r.t. theta 
# def gradient(X, y, theta): 
# 	h = hypothesis(X, theta) 
# 	grad = np.dot(X.transpose(), (h - y)) 
# 	return grad 

# # function to compute the error for current values of theta 
# def cost(X, y, theta): 
# 	h = hypothesis(X, theta) 
# 	J = np.dot((h - y).transpose(), (h - y)) 
# 	J /= 2
# 	return J[0] 

# # function to create a list containing mini-batches 
# def create_mini_batches(X, y, batch_size): 
# 	mini_batches = [] 
# 	data = np.hstack((X, y)) 
# 	np.random.shuffle(data) 
# 	n_minibatches = data.shape[0] // batch_size 
# 	i = 0

# 	for i in range(n_minibatches + 1): 
# 		mini_batch = data[i * batch_size:(i + 1)*batch_size, :] 
# 		X_mini = mini_batch[:, :-1] 
# 		Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
# 		mini_batches.append((X_mini, Y_mini)) 
# 	if data.shape[0] % batch_size != 0: 
# 		mini_batch = data[i * batch_size:data.shape[0]] 
# 		X_mini = mini_batch[:, :-1] 
# 		Y_mini = mini_batch[:, -1].reshape((-1, 1)) 
# 		mini_batches.append((X_mini, Y_mini)) 
# 	return mini_batches 

# # function to perform mini-batch gradient descent 
# def gradientDescent(X, y, learning_rate = 0.001, batch_size = 32): 
# 	theta = np.zeros((X.shape[1], 1)) 
# 	error_list = [] 
# 	max_iters = 3
# 	for itr in range(max_iters): 
# 		mini_batches = create_mini_batches(X, y, batch_size) 
# 		for mini_batch in mini_batches: 
# 			X_mini, y_mini = mini_batch 
# 			theta = theta - learning_rate * gradient(X_mini, y_mini, theta) 
# 			error_list.append(cost(X_mini, y_mini, theta)) 
# 	return theta, error_list 

# theta, error_list = gradientDescent(X_train, y_train) 
# print("Bias = ", theta[0]) 
# print("Coefficients = ", theta[1:]) 

# visualising gradient descent 
#plt.plot(error_list) 
#plt.xlabel("Number of iterations") 
#plt.ylabel("Cost") 
#plt.show()
