## ---------------------------------------------------------------------------------------------------
#公众号 遇见YI算法文章：遇见YI算法之逻辑回归 代码
##------------------------------------

------------------------------------------------------------
方法 1：根据逻辑回归原理公式做logit
--------------------------------------
## 例子
from numpy import *
import matplotlib.pyplot as plt

'''
##未做归一化处理
def loadData(filename):
    data = loadtxt(filename)
    m,n = data.shape
    print 'the number of  examples:',m
    print 'the number of features:',n-1
    x = data[:,0:n-1]
    y = data[:,n-1:n]
    return x,y
'''
##做归一化处理	
def loadData(filename):
    data = loadtxt(filename)
    m,n = data.shape
    print ('the number of  examples:',m)
    print ('the number of features:',n-1)
    x = data[:,0:n-1]
    max = x.max(0)
    min = x.min(0)
    x = (x - min)/((max-min)*1.0)     #scaling
    y = data[:,n-1:n]
    return x,y


#the sigmoid function
def sigmoid(z):
    return 1.0 / (1 + exp(-z))


#the cost function
def costfunction(y,h):
    y = array(y)
    h = array(h)
    J = sum(y*log(h))+sum((1-y)*log(1-h))
    return J


# the batch gradient descent algrithm
def gradescent(x,y):
    m,n = shape(x)     #m: number of training example; n: number of features
    x = c_[ones(m),x]     #add x0
    x = mat(x)      # to matrix
    y = mat(y)
    a = 0.0000025       # learning rate
    maxcycle = 4000
    theta = zeros((n+1,1))  #initial theta

    J = []
    for i in range(maxcycle):
        h = sigmoid(x*theta)
        theta = theta + a * (x.T)*(y-h)
        cost = costfunction(y,h)
        J.append(cost)

    plt.plot(J)
    plt.show()
    return theta,cost


#the stochastic gradient descent (m should be large,if you want the result is good)
def stocGraddescent(x,y):
    m,n = shape(x)     #m: number of training example; n: number of features
    x = c_[ones(m),x]     #add x0
    x = mat(x)      # to matrix
    y = mat(y)
    a = 0.01       # learning rate
    theta = ones((n+1,1))    #initial theta

    J = []
    for i in range(m):
        h = sigmoid(x[i]*theta)
        theta = theta + a * x[i].transpose()*(y[i]-h)
        cost = costfunction(y,h)
        J.append(cost)
    plt.plot(J)
    plt.show()
    return theta,cost


#plot the decision boundary
def plotbestfit(x,y,theta):
    plt.plot(x[:,0:1][where(y==1)],x[:,1:2][where(y==1)],'ro')
    plt.plot(x[:,0:1][where(y!=1)],x[:,1:2][where(y!=1)],'bx')
    x1= arange(-4,4,0.1)
    x2 =(-float(theta[0])-float(theta[1])*x1) /float(theta[2])

    plt.plot(x1,x2)
    plt.xlabel('x1')
    plt.ylabel(('x2'))
    plt.show()


def classifyVector(inX,theta):
    prob = sigmoid((inX*theta).sum(1))
    return where(prob >= 0.5, 1, 0)


def accuracy(x, y, theta):
    m = shape(y)[0]
    x = c_[ones(m),x]
    y_p = classifyVector(x,theta)
    accuracy = sum(y_p==y)/float(m)
    return accuracy

##调用上面代码：
x,y = loadData("E:/Python/work/horseColicTraining.txt")
theta,cost = gradescent(x,y)
print ('J:',cost)

ac_train = accuracy(x, y, theta)
print ('accuracy of the training examples:', ac_train)

x_test,y_test = loadData('E:/Python/work/horseColicTest.txt')
ac_test = accuracy(x_test, y_test, theta)
print ('accuracy of the test examples:', ac_test)


------------------------------------------------------------
方法 2：使用sklearn模块做logit
-------------------------------
##引入包使用
import scipy as sp  
import numpy as np  
from sklearn.cross_validation import train_test_split  
from sklearn import metrics  
from sklearn.linear_model import LogisticRegression  

##定义引入数据并归一化处理
def loadData(filename):
    data = loadtxt(filename)
    m,n = data.shape
    print ('the number of  examples:',m)
    print ('the number of features:',n-1)
    x = data[:,0:n-1]
    max = x.max(0)
    min = x.min(0)
    x = (x - min)/((max-min)*1.0)     #scaling
    y = data[:,n-1:n]
    return x,y
	
##引入训练数据并归一化处理
x,y = loadData("E:/Python/work/horseColicTraining.txt")
x_train=x
##将y内list合并
y1=[]
for i in range(len(y)):
	y1.extend(y[i].tolist())
y_train=np.array(y1)
#print(x_train)
#print(y_train)     #查看样本 

#调用逻辑回归  
model = LogisticRegression()  
model.fit(x_train, y_train)  
print(model)       #输出模型  

##引入测试数据并归一化处理
x,y = loadData("E:/Python/work/horseColicTest.txt")
x_test=x
##将y内list合并
y1=[]
for i in range(len(y)):
	y1.extend(y[i].tolist())
y_test=np.array(y1)
#print(x_test)
#print(y_test)     #查看样本

# 预测make predictions  
expected = y_test                       #测试样本的期望输出  
predicted = model.predict(x_test)       #测试样本预测 

#输出结果  
print(metrics.classification_report(expected, predicted))       #输出结果，精确度、召回率、f-1分数  
print(metrics.confusion_matrix(expected, predicted))            #混淆矩阵 
