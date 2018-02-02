import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    #print (y.size)
    d = X.shape[1] # d value
    k_class_matrix = np.unique(y)
    #print (k_class_matrix)
    k_class = np.shape((np.unique(y)))[0]
    #print (k_class)
    means = np.zeros((d,k_class)) #create means matrix
    for i in range(1,k_class+1):
        means[:,i-1] = np.mean(X[np.where(y==i)[0]].T,axis=1) #calculates mean for each class
    #print (means)
    covmat = np.cov(X.T)   #calculates covariance for pooled data
    #print (covmat)
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    #print(np.shape(X))
    #print(np.shape(y))
    #print(X)
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes

    # IMPLEMENT THIS METHOD

    prediction_classes = len(np.unique(y))
    d = np.shape(X)[1]
    means = np.zeros((d, prediction_classes))
    covmats = []
    for i in range(1, prediction_classes+1):
        indexes = np.where(y.flatten()==i)[0]
        #print(indexes)
        indexes = list(indexes)
        selectData = X[indexes, :] # All the N(150) data is split into D * K Classes
        #print(np.shape(selectData))
        selectDataTransposed = selectData.T # 2 * Variable number of values
        calculatedMean = np.mean(selectDataTransposed, axis=1)  #2 * Mean Values
        means[:, i-1] = calculatedMean
        #print(np.cov(selectDataTransposed)) #2 * 2 Matrix
        covmats.append(np.cov(selectDataTransposed))
        #print(np.shape(means))
    #print(means)
    #print(covmats)
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    k_classes = means.shape[1]
    N = Xtest.shape[0]
    d = Xtest.shape[1]
    #print (Xtest.shape)
    # print (k_classes)
    #print (X_input)
    ypred = np.zeros((ytest.shape))  #ypred array initialise to zeros in the same size as ytest
    #print (ypred)
    pdf_value = np.zeros((N,k_classes))

    for xinput in range(N):
         for k in range(k_classes):
            expterm = np.dot(((Xtest[xinput] - means[:,k].T).T),np.dot(inv(covmat),Xtest[xinput] - means[:,k].T))
            inter_expterm = -expterm/2
            final_expterm = np.exp(inter_expterm)
            covar_det = np.sqrt(det(covmat))
            denom = pow(2*pi,d/2.0) * covar_det
            pdf_value[xinput,k] = final_expterm/denom
    #print (pdf_value)
    max_value = np.argmax(pdf_value,1)+1 #get the maxvlaues indices positions
    #print (max_value.shape)
    ypred = np.array(max_value)
    ypred = ypred[np.newaxis] #adds new axis so as to compare the ypred and ytest values
    ypred = ypred.T
    count = 0
    for i in range(N):         #labels matching the actual values
        if ypred[i]==ytest[i]:
           count = count + 1
    acc = count/N * 100
    #print(ypred)
    #print(acc)

    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels
    # IMPLEMENT THIS METHOD
    prediction_classes = np.shape(means)[1]
    inputs = np.shape(Xtest)[0]
    sumofcorrect = 0
    ytest = ytest.astype(int)
    ypred = np.zeros(np.shape(ytest))
    #print("Y Test shape " , np.shape(ytest))
    #print(Xtest[0,:])
    for input in range(1, inputs + 1):
        predictpdf = []
        currentRecord = np.transpose(Xtest[input-1,:])
        #print("Means", means)
        for i in range(1, prediction_classes + 1):
            predictclass = 0
            d = np.shape(covmats[i-1])[0]
            detMatrix = np.linalg.det(covmats[i-1]) # Determinant Matrix of Covariance - E
            invMatrix = np.linalg.inv(covmats[i-1]) # Inverse Matrix of Covariance
            #print("QDA Actual Inv", invMatrix)
            #print("QDA Actual Det", detMatrix)
            #print(np.shape(currentRecord))
            means_product_transpose = np.dot((currentRecord - means[:, i-1]).T,invMatrix)
            #print("Means Product Transpose " , np.shape(means_product_transpose))
            means_product = currentRecord - means[:, i-1]
            #print("Means Product" , np.shape(means_product))
            predictpdf.append(np.exp((-0.5) * np.dot(means_product_transpose, means_product)) / np.multiply((np.power(np.pi*2, d/2)) , np.power((detMatrix), 0.5)))
            #print("Predict PDF", predictpdf)
        max_value = max(predictpdf) # Gets the Max value from the predictedPDF
        index = predictpdf.index(max_value) + 1 # Gets the Index that contains the maximum PDF
        np.put(ypred, [input-1], index)
        if (index == ytest[input-1]):
            sumofcorrect = sumofcorrect + 1 # Running sum of correct predicitions
    acc = sumofcorrect/np.shape(Xtest)[0] * 100
    #print("Accuracy", acc)
    #print("Shape of Prediciton", np.shape(ypred))
    #print(ypred)
    return acc,ypred
def learnOLERegression(X,y):
    # Inputs:
    # X = N x d
    # y = N x 1
    # Output:
    # w = d x 1
    # X - 242 * 64
    # y - 242 * 1
    # w - 64 * 1
    #print(X[1])
    #print(np.shape(X)[0])
    #print(np.shape(X))
    #print(np.shape(y))
    # IMPLEMENT THIS METHOD
    w = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,y))
    #print(np.shape(w))
    #print(w)
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d
    # y = N x 1
    # lambd = ridge parameter (scalar)
    #X_transpose = X.T
    Identity_mat = np.identity(X.shape[1])
    X_lam = np.dot(X.T,X) + lambd * Identity_mat
    w = np.linalg.solve(X_lam,np.dot(X.T,y))
    #print(w)
    # Output:
    # w = d x 1
    # IMPLEMENT THIS METHOD
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    #print(np.shape(Xtest))
    #print(np.shape(ytest))
    #print(np.shape(w))
    mse = np.sum(np.square((ytest - (np.dot(Xtest, w))))) / np.shape(Xtest)[0]
    #print(mse)
    #print(sum1)
    # IMPLEMENT THIS METHOD
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda
    w=w.reshape(X.shape[1],1)
    N = X.shape[0]
    y_X = y-np.dot(X,w)
    y_X_t = np.dot(y_X.T,y_X)
    w_t = np.dot(w.T,w)
    error = (0.5 * y_X_t) + (0.5 * lambd * w_t)
    #error_grad = ((-1.0) * np.dot(y.T,X) + np.dot(w.T,np.dot(X.T,X))+(lambd * w.T))/N
    error_grad = (((np.dot(w.T, np.dot(X.T, X)) - np.dot(y.T, X))) + (w.T * lambd))
    error_grad = error_grad.flatten()
    # IMPLEMENT THIS METHOD
    #print(error)
    #print(error_grad)
    return error, error_grad


def mapNonLinear(x,p):
    # Inputs:
    # x - a single column vector (N x 1)
    # p - integer (>= 0)
    # Outputs:
    # Xd - (N x (d+1))
    #N = x.shape[0]
    #initializing xd as an np array of size N*p+1
    #Xd = np.zeros((N,p+1))
    #for i in range(0,p+1):
    #    Xd[:, i] = x ** i
    # IMPLEMENT THIS METHOD
    Xd = np.zeros((x.shape[0],p+1))
    for i in range(0,p+1) :
        Xd[:,i] = pow(x,i)

    return Xd

# Main script

# Problem 1
# load the sample data
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()
#Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

#add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)
print("Mean Value of Weights obtained in Linear Regression " + str(np.mean(w_i)))

#Training Data Calculation
mle_training = testOLERegression(w,X,y)
mle_training_intercept = testOLERegression(w_i,X_i,y)

#print(w_i)
plt.plot(w_i)
plt.ylabel('Weight Maginitude of Linear Regression')
plt.show()
print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))
print('MSE on Training Data without intercept' + str(mle_training))
print('MSE on Training Data with intercept' + str(mle_training_intercept))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    #Lambda of 0.06 is the optimal value
    if(lambd == 0.06):
        plt.plot(w_l)
        plt.ylabel('Weight Maginitude of Ridge Regression')
        plt.show()
        print("Mean Value of Weights obtained in Ridge Regression " + str((np.mean(w_l))))
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1

#print(w_l)
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 60}    # Preferred value.
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(mses3)]
print (lambda_opt) # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
