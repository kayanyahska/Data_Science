# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt


def cost_function(X, Y, theta,lem):
    m = Y.shape[0]
    J = np.sum((np.dot(X,theta) - Y) ** 2)/(2 * m)
    J=J+lem*np.sum(theta**2)
    return J

def gradient_descent(X, Y,alpha,itr,lem):
    theta=np.ones(X.shape[1])
    theta=np.reshape(theta,(-1,1))
    m=Y.shape[0]
    cost=[]
    for i in range(itr):
        theta=theta*(1-lem*(alpha/m)) - (alpha/m)*(np.dot(X.T,np.dot(X,theta)-Y))
        cost.append(cost_function(X, Y, theta,1))
    return (theta,cost)

def predict_value(X,theta):
    return np.dot(X,theta)

def Cost_Vs_Itr(cost,itr,alpha,lem):
    I=[i for i in range(itr)]
    for i in range(len(alpha)):
        plt.plot(I,cost[i],label=alpha[i])
        plt.title("lem = " +str(lem))
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()

def RMSE(Y, Y_pred):
    return np.sqrt(sum((Y - Y_pred) ** 2) / len(Y))

def print_theta(alpha,theta):
    for i in range(len(alpha)):
        print("theta for alpha = ",alpha[i],sep=' ')
        print(theta[i])


# Read From csv File
csv=sys.argv[1]
data=pd.read_csv(csv)
m=len(data.columns)

# Store in list
X=data.iloc[:,0:m-1].values
Y=data.iloc[:,m-1:m].values



# Normalise the data
X=(X-np.mean(X))/np.std(X)
Y=(Y-np.mean(Y))/np.std(Y)

# Initialize indices
train_start=0
train_end=int(0.8*len(Y))

test_start=int(0.8*len(Y))
test_end=len(Y)

# Segregate into Training data and testing data
X_train=np.array(X[train_start:train_end])
Y_train=np.array(Y[train_start:train_end])
X_test=np.array(X[test_start:test_end])
Y_test=np.array(Y[test_start:test_end])
alpha=[0.01,0.001,0.0001,0.00001]
itr=1000



# Without Regularisation
def Linear_Regression():
    print("-------------------------------------------------Before Regularisation-------------------------------------------------")
    # Regress
    Cost=[]
    Theta=[]
    for i in range(len(alpha)):
        theta,cost=gradient_descent(X_train,Y_train,alpha[i],itr,0)
        Cost.append(cost)
        Theta.append(theta)

    Cost=np.array(Cost)
    Theta=np.array(Theta)

    print_theta(alpha,Theta)

    # Cost Vs Iterations
    Cost_Vs_Itr(Cost,itr,alpha,0)

    # Predicted value of Y
    Y_pred=predict_value(X_test,theta)

    # Model Evaluation

    # RMSE
    rmse=RMSE(Y_test,Y_pred)
    print("RMSE :",rmse,sep='\n')


# After Regularisation
def Ridge_Regression(lem):
    print("-------------------------------------------------After Regularisation-------------------------------------------------")
    print("lambda =",lem)
    # Regress
    Cost=[]
    Theta=[]
    for i in range(len(alpha)):
        theta,cost=gradient_descent(X_train,Y_train,alpha[i],itr,lem)
        Cost.append(cost)
        Theta.append(theta)

    Cost=np.array(Cost)
    Theta=np.array(Theta)

    print_theta(alpha,Theta)

    # Cost Vs Iterations
    Cost_Vs_Itr(Cost,itr,alpha,lem)

    # Predicted value of Y
    Y_pred=predict_value(X_test,theta)

    # Model Evaluation

    # RMSE
    rmse=RMSE(Y_test,Y_pred)
    print("RMSE :",rmse,sep='\n')


Linear_Regression()

lem=[1,10,100,1000]
for i in lem: 
    Ridge_Regression(i)


#---------------------------------------Contour Plots for regularized linear regression
theta1=np.ones(X.shape[1])
theta1=np.reshape(theta1,(-1,1))
#theta1=np.zeros((2,1))
t1=np.linspace(-10,10,X_test.shape[0])
t2=np.linspace(-10,10,X_test.shape[0])

x,y=np.meshgrid(t1,t2)

for p in lem:
  Z=[]
  for i in range(X_test.shape[0]):
    for j in range(X_test.shape[0]):
      theta1[0][0]=x[i][j]
      theta1[1][0]=y[i][j]
      #hyp=hypothesis(theta1,X_test)
      cost=cost_function(X_test, Y_test, theta1,p)
      Z.append(cost)

  fig,ax=plt.subplots(1,1)
  Z=np.array(Z)
  Z=Z.reshape(X_test.shape[0],X_test.shape[0])
  cp=ax.contourf(x,y,Z)
  fig.colorbar(cp)
  title="Contour Plot for regularized regression with lambda = "+str(p) +"\n"
  ax.set_title(title)
  ax.set_xlabel('theta 1')
  ax.set_ylabel('theta 2')
  plt.show()
