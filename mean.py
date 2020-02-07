#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt

def prior(mu):
    n = len(mu)
    prior = np.ones(n)
    return prior

def like(data,sigma,mu):
    P = 1.0
    for j in sigma:
        for i in data:
            P *= (1/(j*np.sqrt(2*np.pi)))*np.exp(-((i-mu)**2/(2*j**2)))*prior(mu)
    
    return P/np.trapz(P,mu)

def maximo_sigma(x, y):
    deltax = x[1] - x[0]

    # maximo de y
    ii = np.argmax(y)

    # segunda derivada
    d = (y[ii+1] - 2*y[ii] + y[ii-1]) / (deltax**2)

    return x[ii], 1.0/np.sqrt(-d)

mu = np.linspace(-5,10,100)
x = np.array([4.6, 6.0, 2.0, 5.8])
sigma = np.array([2.0, 1.5, 5.0, 1.0])

post = like(x[:4],sigma,mu)

#sigma = 5


#max = np.argmax(like(x[:2],sigma,mu))

max, sigma_f = maximo_sigma(mu, np.log(post))



plt.figure()
plt.plot(mu,like(x[:4],sigma,mu))
plt.ylabel(r'$P(x_k \vert mu)$')
plt.xlabel(r'$\mu$')
plt.title(r'${:.2f}\pm {:.2f}$'.format(max,sigma_f))
plt.savefig('mean.png')


# In[ ]:




