import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sample_data = [
        [-0.017612, 14.053064, 0],
        [-1.395634, 4.662541, 1],
        [-0.752157, 6.538620, 0],
        [-1.322371, 7.152853, 0],
        [0.423363, 11.054677, 0],
        [0.406704, 7.067335, 1]
    ]

# training set
def loadDataSet(sample):
    dataMat = []
    labelMat = []
    for line in sample:
        lineArr = line
        #   features,x0=1
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        #   target variables
        labelMat.append(int(lineArr[-1]))
    return dataMat,labelMat

# sigmoid function
def sigmoid(inX) :
    return 1/(1+np.exp(-inX))

def gradAscent(dataMatIn,classLabels):
    # (m,n)
    dataMatrix = np.mat(dataMatIn)
    # 转置后(m,1)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(dataMatrix)
    # 定义初始化回归系数(n,1)
    weights = np.ones((n,1))
    # 定义梯度下降步长
    alpha = 0.001
    # 定义最大循环次数
    maxCycles = 500
    for i in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        # 梯度上升
        weights = weights + alpha *  dataMatrix.transpose() * error
    return weights

dataArr,labelMat = loadDataSet(sample_data)
weights = gradAscent(dataArr,labelMat)
# 惊呆了，得出的结果居然不一样

def plotBestFit(weights):
    dataMat,labelMat=loadDataSet(sample_data)
    n=np.shape(dataMat)[0]
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(n):
        if labelMat[i]==1:
            xcord1.append(dataMat[i][1])
            ycord1.append(dataMat[i][2])
        else:
            xcord2.append(dataMat[i][1])
            ycord2.append(dataMat[i][2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=np.arange(-3,3,0.1)
    y=(-weights[0,0]-weights[1,0]*x)/weights[2,0] #matix
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

plotBestFit(weights)