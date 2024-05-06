# Compressed sensing solve with ISTA, FISTA and AFBN in python

#Reference:
# for math perspective
# Beck, Amir, and Marc Teboulle. "A fast iterative shrinkage-thresholding algorithm for linear inverse problems." SIAM journal on imaging sciences 2.1 (2009): 183-202.
# for compressed sensing understanding
# https://humaticlabs.com/blog/compressed-sensing-python/
# for a ISTA and FISTA algorithm in python
# https://gist.github.com/agramfort/ac52a57dc6551138e89b

import scipy.fftpack as spfft
import matplotlib.pyplot as plt
import numpy as np

def soft_thresh(x, l):
    return np.sign(x) * np.maximum(np.abs(x) - l, 0.)

def ista(A, b, l, maxit):
    costs=[]
    x = np.zeros(A.shape[1])
    L = np.linalg.norm(A) ** 2 /2 # Lipschitz constant
    for i in range(maxit):
        x = soft_thresh(x + np.dot(A.T, b - A.dot(x)) / L, l / L)
        costs.append(np.linalg.norm(A.dot(x)-b)**2 + l* np.linalg.norm(x,1))
    return x,costs

def fista(A, b, l, maxit):
    x = np.zeros(A.shape[1])
    t = 1
    z = x.copy()
    L = np.linalg.norm(A) ** 2 /2
    costs = []
    for i in range(maxit):
        xold = x.copy()
        z = z + A.T.dot(b - A.dot(z)) / L
        x = soft_thresh(z, l / L)
        t0 = t
        t = (1. + np.sqrt(1. + 4. * t ** 2)) / 2.
        z = x + ((t0 - 1.) / t) * (x - xold)
        costs.append(np.linalg.norm(A.dot(x) - b) ** 2 + l * np.linalg.norm(x, 1))
    return x,costs

def afbn(A, b, l, maxit,alpha=4.):
    x = np.zeros(A.shape[1])
    z = x.copy()
    L = np.linalg.norm(A) ** 2 /2
    costs=[]
    for i in range(maxit):
        xold = x.copy()
        z = z + A.T.dot(b - A.dot(z)) / L
        x = soft_thresh(z, l / L)
        z = x + (i - 1.) / (i+alpha-1) * (x - xold)
        costs.append(np.linalg.norm(A.dot(x) - b) ** 2 + l * np.linalg.norm(x, 1))
    return x,costs


n=5000
l=1/8
t=np.linspace(0,l,n)
dt=t[1]-t[0]

#defining the signal
x = np.sin(1394* np.pi * t ) + np.cos(3266 * np.pi * t)

#Defining the sampling
perc = 10
k = round( perc/100 * n)
ri = np.random.choice(n,k) #random indices
b = x[ri]


xf = spfft.dct(x) #obtain the frequency representaion from the signal
dfs = xf * np.conj(xf)/n #spectra density
fr = np.arange(n)* (1 / l) #freqquency resolution

plt.plot(fr[:500],dfs[:500])

A = spfft.idct(np.identity(n), norm='ortho', axis=0) #Defining the operator A
A = A[ri] #Defining PHI A, the operetator sampled

niter=400
l=0.2

sista,costista=ista(A,b,l,niter)
xista=spfft.idct(sista,norm='ortho',axis=0)

sfista,costsfista=fista(A,b,l,niter)
xfista=spfft.idct(sfista,norm='ortho',axis=0)

alphas=[3.,10.,100.,200.]
varafbn=[]
for alpha in alphas:
    safbn, costsafbn = afbn(A, b, l, niter,alpha)
    xafbn = spfft.idct(safbn, norm='ortho', axis=0)
    varafbn.append([safbn,costsafbn,xafbn])

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(t, x, t[ri],b,'r.')
axs[0,0].set_xlim(0,.012)
axs[0, 0].set_title(f'Amostra Sinal {perc}%')
axs[0, 1].plot(t, x, t, xista, 'r')
axs[0,1].set_xlim(0,.012)
axs[0, 1].set_title('ISTA')
axs[1, 0].plot(t, x, t, xfista, 'r')
axs[1,0].set_xlim(0,.012)
axs[1, 0].set_title('FISTA')
axs[1, 1].plot(t, x, t, varafbn[0][2], 'r')
axs[1,1].set_xlim(0,.012)
axs[1, 1].set_title('AFBN')
fig.suptitle(f'Comparação Métodos niter = {niter}')

iter=range(niter)

fig,axs=plt.subplots()
axs.plot(iter,costista,iter,costsfista)
for i in range(4):
    axs.plot(iter,varafbn[i][1])
axs.legend(['ISTA','FISTA',
            f'AFBN alfa = {alphas[0]}',
            f'AFBN alfa = {alphas[1]}',
            f'AFBN  alfa = {alphas[2]}',
            f'AFBN alfa = {alphas[3]}'])
axs.set_xlabel('Iterações')
axs.set_ylabel('Função Custo')
plt.show()
