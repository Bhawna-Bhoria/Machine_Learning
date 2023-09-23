import numpy as np
import pandas as pd
import io
from google.colab import files
files =files.upload()

import matplotlib.pyplot as plt
org_data =pd.read_csv(io.BytesIO(files['mnist_test.csv']))
org_data.head()
org_data = np.array(org_data)
x,y= org_data.shape
data = org_data[0:1000]
Y_train = data.T[0]
X_train = data.T[1:y]

test_data = org_data[1000:x].T
X_test = test_data[1:y]
Y_test = test_data[0]


def initialisation(input_nodes, hidden_nodes):
  output_nodes = len(set(Y_train)) #no. of classes
  print(output_nodes)
  W1 = np.random.rand(hidden_nodes, input_nodes) - 0.5
  b1 = np.random.rand(hidden_nodes, 1) - 0.5
  W2 = np.random.rand(output_nodes, hidden_nodes) - 0.5
  b2 = np.random.rand(output_nodes, 1) - 0.5
  return W1, b1, W2, b2

def sigmoid_act(Z):
  temp = 1 + np.exp(-Z)
  value = 1 / temp
  return value

def deriv_sigmoid(Z):
  temp = sigmoid_act(Z)
  value = temp * (1 - temp)
  return value

def softmax(Z):
    temp = np.exp(Z - Z.max())
    value = temp / np.sum(temp, axis = 0)
    return value


def f_prop(W1, b1, W2, b2, X):
  Z1 = np.matmul(W1, X) + b1
  A1 = sigmoid_act(Z1)
  Z2 = W2.dot(A1) + b2
  A2 = softmax(Z2)
  return Z1, A1, Z2, A2

def one_hot(Y):
  one_hot_Y = np.zeros((Y.size, Y.max() + 1))
  m,n = one_hot_Y.shape
  one_hot_Y[np.arange(Y.size), Y] = 1
  return one_hot_Y.T

def get_predictions(A2):
  value = np.argmax(A2,0) 
  return value

def get_accuracy(predictions,Y):
  #print(predictions, Y)
  size = Y.size
  value = np.sum(predictions == Y) /size
  return value

def crossentropy_loss(output_prob, Y):
    s = np.sum(np.multiply(Y, np.log(output_prob)))
    m = Y.shape[0]
    value = -(1./(m * 10000)) * s 
    return value

def update_param( W1, b1, W2, b2, dW1, db1, dW2, db2, lr):
  tempdW1 = lr*dW1
  W1 = W1 - tempdW1
  tempdb1 = lr*db1
  b1 = b1 - tempdb1
  tempdW2 = lr*dW2
  W2 = W2 - tempdW2
  tempdb2 = lr*db2
  b2 = b2 - tempdb2
  return W1,b1,W2,b2

def b_prop(Z1, A1, Z2, A2, W2, X, Y):
  one_hot_Y = one_hot(Y)
  dZ2 = A2 - one_hot_Y
  # dW2 = 1/m * dZ2.dot(A1.T)
  # db2 = 1/m * np.sum(dZ2,2)
  # dZ1 = W2.T.dot(dZ2) * deriv_sigmoid(Z1)
  # dW1 = 1/m * dZ2.dot(X.T)
  # db1 = 1/m * np.sum(dZ1,2)
  tempdw2 = np.matmul(dZ2, A1.T)
  dW2 = 1/ Y.size * tempdw2
  tempdb2 = np.sum(dZ2, axis = 1, keepdims = True)
  db2 = 1/ Y.size * tempdb2
    
  dA1 = np.matmul(W2.T, dZ2)
  dZ1 = dA1 * deriv_sigmoid(Z1)
  tempdw1=np.matmul(dZ1, X.T)
  dW1 = 1/ Y.size * tempdw1
  tempdb1 = np.sum(dZ1, axis = 1, keepdims = True)
  db1 = 1/ Y.size * tempdb1
  return dW1, db1, dW2, db2,one_hot_Y

'''

#Function to find value of expression at a given point
def find_value(expr,point):
  return np.float64([_.subs({u:point[0], v:point[1]}) for _ in expr][0])
epsilon=0.06

#Funtion to find the gradient
def find_gradient(expr,point):
  deriv_u=[Derivative(expr, u).doit()]
  deriv_v=[Derivative(expr, v).doit()]
  gradient=np.array([find_value(deriv_u,point),find_value(deriv_v,point)])
  return gradient

alpha=1
gfx0=find_gradient(y,x0)
gfx0

#To initiate the loop
lhs=1
rhs=0


x1=x0
while(lhs>rhs):
    alpha=alpha/2
    x0=x1
    d=find_gradient(y,x0)
    x1=np.add(x0,-alpha*d)
    fx1=find_value([y],x1)
    #Armijo Condition lhs
    lhs=fx1
    fx0=find_value([y],x0)
    gfx0=find_gradient(y,x0)
    gfx1=find_gradient(y,x1)
    n=len(gfx0)
    A=np.zeros((n,0),dtype=float)
    A=np.column_stack((A,gfx0))
    B= np.float64(np.matmul(A.T,d))
    #Armijo Condition rhs
    rhs=fx0-alpha*beta1*B
    gfx1 = np.array(gfx1, dtype=np.float64)
    vec_norm = float(np.linalg.norm(alpha))
    if(vec_norm<epsilon):
      break
print(f" Alpha is {alpha}")

'''
def gradient_descent_train(X,Y,iterations, alpha,W1,b1,W2,b2):
  #W1,b1,W2,b2 = init_params()
  #tloss = 0
  loss = 0
  loss_list = []
  accuracy_list = []
  print(W1,b1,W2,b2)
  for i in range(iterations):
    
    Z1,A1,Z2,A2 =  f_prop(W1, b1, W2, b2, X)
    dW1, db1, dW2, db2,one_hot_encoded_Y = b_prop(Z1,A1,Z2,A2,W2,X,Y)
    W1,b1,W2,b2 = update_param(W1,b1,W2,b2,dW1,db1,dW2,db2,alpha)

    predicted_val = get_predictions(A2)
    accuracy=get_accuracy(predicted_val, Y)
    #tloss = loss
    loss = crossentropy_loss(A2, one_hot_encoded_Y)
    loss_list.append(loss)
    accuracy_list.append(accuracy)
    # if i % 50 == 0:
    #   print("Iteration: ", i)
    #   print("Accuracy: ", get_accuracy(get_predictions(A2),Y))
    if i == iterations -1:
      print(("Accuracy = {}, Loss = {}").format(i, accuracy, loss))
      print(("Epoch {}: Accuracy = {}, Loss = {}").format(i, accuracy, loss))
  return W1,b1,W2,b2,A2,accuracy_list,loss_list




H_list = [10, 10, 20, 2, 4]
T_list = [0.01, 0.1, 0.02, 0.01, 0.5]
E_list = [20, 50, 50, 10, 25]

N = 784 

for b in range(5):
    W1,b1,W2,b2 = initialisation(N, H_list[b])
    print(("Epoch = {}, N = {}, H = {}, learning_rate = {}").format(E_list[b], N, H_list[b], T_list[b]))
    W1, b1, W2, b2, A2, accuracy_list, loss_list = gradient_descent_train(X_train, Y_train, E_list[b], T_list[b], W1, b1, W2, b2)
    print("Configuration Set ", b+1)
    print("W1 is",W1)
    print("b1 is",b1)
    print("W2 is",W2)
    print("b2 is",b2)
    
    plt.plot(range(E_list[b]), accuracy_list)
    plt.title("Accuracy vs Epoch")
    plt.show()
    plt.plot(range(E_list[b]), loss_list)
    plt.title("Loss vs Epoch")
    plt.show()
    
    
    
    

