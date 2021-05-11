import numpy as np
import matplotlib.pyplot as pl


# Define the kernel function
def kernel(a, b, param):
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/param) * sqdist)
param = 0.1


# Define the acquisition function
def acquisition(e,q) :
    y= e+1.5*q
    t=[]
    for s in range (0,49):
        if y[s]<0:
            t.append(0)
        else:
            t.append(y[s])
    return t  


# Define the function that plot the acquisition function
def plot_aqci(h,o):
    p = pl.plot(h,o)
    pl.show()
    return p


# Define the function that help us to find
# the absciss of the next point
def explo_point(f,j):
    z = max(acquisition(f,j))
    c = 0
    for i in acquisition(f,j):
        if i != z:
            c = c+1
        else:
            break
    b = np.linspace(4, 12, 49)[c]
    return b


# Test data
n = 50
Xtest = np.linspace(4, 12, n).reshape(-1,1)
K_ss = kernel(Xtest, Xtest, param)


# Training data
H = [4,6,7,9.5,11,12]
G=np.array(H)
Xtrain = G.reshape(6,1)

# Optimisation loop
d=1
l=0
v=1

while v!= l :
    ytrain = -1*np.cos(np.exp(Xtrain)+3)

    # Apply the kernel function to our training points
    K = kernel(Xtrain, Xtrain, param)
    L = np.linalg.cholesky(K + 0.00005*np.eye(len(Xtrain)))

    # Compute the mean at our test points
    K_s = kernel(Xtrain, Xtest, param)
    Lk = np.linalg.solve(L, K_s)
    mu = np.dot(Lk.T, np.linalg.solve(L, ytrain)).reshape((n,))

    # Compute the standard deviation so we can plot it
    s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
    stdv = np.sqrt(s2)

    # Graphs
    pl.plot(Xtrain, ytrain, 'bs', ms=8)
    pl.gca().fill_between(Xtest.flat, mu-1.5*stdv, mu+1.5*stdv, color="#dddddd")
    pl.plot(Xtest, mu, 'r--', lw=2)
    pl.show()

    # Acquisition function graph
    plot_aqci(np.linspace(4, 12, 49), acquisition(mu,stdv))


    # Exploration point
    b = explo_point(mu,stdv)
    H.append(b)
    G = np.array(H)
    Xtrain = G.reshape(6+d,1)

    d = d+1
    v = l
    l = b
    
# The optimal function
pl.plot(Xtest,mu,'r--')
pl.show()