import numpy as np
from ham import *
import warnings

class mps:


    def __init__(self,site_dimension=2,bond_dimension=100):
        self.site_dimension = site_dimension
        self.bond_dimension = bond_dimension
        self.product_state()




    @property
    def site_dimension(self):
        return self._site_dimension

    @property
    def bond_dimension(self):
        return self._bond_dimension

    @property
    def Gamma(self):
        return self._Gamma

    @property
    def Lambda(self):
        return self._Lambda

    @site_dimension.setter
    def site_dimension(self, d):
        self._site_dimension=d

    @bond_dimension.setter
    def bond_dimension(self, chi):
        self._bond_dimension=chi

    @Gamma.setter
    def Gamma(self,gamma):
        self._Gamma=gamma

    @Lambda.setter
    def Lambda(self,lambd):
        self._Lambda=lambd

    def product_state(self,neel=False):
        """neel=True to inizialize as a neel state"""
        self.Gamma=[]
        self.Lambda=[]
        d=self.site_dimension
        for i in range(2):
                self.Lambda.append(np.ones([1]))
                self.Gamma.append(np.zeros([d,1,1]))
        if neel:
                self.Gamma[0][0,0,0]=1
                self.Gamma[1][d-1,0,0]=1
        else:
                self.Gamma[0][0,0,0]=1
                self.Gamma[1][0,0,0]=1

    def evol(self,U):
        d=self.site_dimension
        chi=self.bond_dimension
        U=np.reshape(U,(d,d,d,d))
        for i_bond in [0,1]:
            ia = np.mod(i_bond-1,2); ib = np.mod(i_bond,2); ic = np.mod(i_bond+1,2)
            chia = self.Gamma[ib].shape[1]; chic = self.Gamma[ic].shape[2]
            # Construct theta matrix and time evolution #
            theta = np.tensordot(self.Gamma[ib],self.Gamma[ic],axes=(2,1)) # i a j b
            theta = np.tensordot(U,theta,axes=([2,3],[0,2])) # ip jp a b 
            theta = np.tensordot(np.diag(self.Lambda[ia]),theta,axes=([1,2])) # a ip jp b 
            theta = np.reshape(np.transpose(theta,(1,0,2,3)),(d*chia,d*chic)) # ip a jp b
            # Schmidt decomposition #
            X, Y, Z = np.linalg.svd(theta,full_matrices=0)
            chi2 = np.min([np.sum(Y*Y/max(Y)**2>10.**(-10)), chi]) #provo a tagliare sullo spettro, quindi su Y^2
            if chi2==chi:
                warnings.warn('Entanglement is growing a lot')
            piv = np.zeros(len(Y), np.bool)
            piv[(np.argsort(Y)[::-1])[:chi2]] = True
            Y = Y[piv]; invsq = np.sqrt(sum(Y**2))
            X = X[:,piv] 
            Z = Z[piv,:]
            # Obtain the new values for self.Gamma and s #
            self.Lambda[ib] = Y/invsq 
            X=np.reshape(X,(d,chia,chi2))
            X = np.transpose(np.tensordot(np.diag(self.Lambda[ia]**(-1)),X,axes=(1,1)),(1,0,2))
            self.Gamma[ib] = np.tensordot(X, np.diag(self.Lambda[ib]),axes=(2,0))
            self.Gamma[ic] = np.transpose(np.reshape(Z,(chi2,d,chic)),(1,0,2))

    def expectation_Sz(self):
        d=self.site_dimension
        sz=np.diag([Sz(conf,0) for conf in range(0,d)])
     #   sz=np.array([[0,1],[1,0]])
        mag=0.
        for i_bond in range(2):
            sB = np.tensordot(np.diag(self.Lambda[np.mod(i_bond-1,2)]),self.Gamma[i_bond],axes=(1,1))
            C=np.tensordot(sB,np.conj(sB),axes=([0,2],[0,2]))
            mag += np.real(np.tensordot(C,sz,axes=([0,1],[0,1])))
        return 0.5*mag
    
    def expectation_SzSz(self,dist,connected=True):
        d=self.site_dimension
        sz=np.diag([Sz(conf,0) for conf in range(0,d)])
        corr=0.
        if dist == 0:
            sz2= np.tensordot(sz,sz,axes=(1,0))
            for i_bond in range(2):
                sB = np.tensordot(np.diag(self.Lambda[np.mod(i_bond-1,2)]),self.Gamma[i_bond],axes=(1,1))
                C=np.tensordot(sB,np.conj(sB),axes=([0,2],[0,2]))
                corr +=np.real( np.tensordot(C,sz2,axes=([0,1],[0,1]))
                        - np.tensordot(C,sz,axes=([0,1],[0,1]))*np.tensordot(C,sz,axes=([0,1],[0,1])))
            return 0.5*corr

        if dist != 0:
            dist=np.abs(dist)
            for i_bond in range(2):
                sB = np.tensordot(np.diag(self.Lambda[np.mod(i_bond-1,2)]),self.Gamma[i_bond],axes=(1,1))
                C=np.tensordot(sB,np.conj(sB),axes=(0,0))
                R = np.tensordot(C,sz,axes=([0,2],[0,1]))
                mean1= np.trace(R)
                for i in range(dist-1):
                    T=np.tensordot(R,self.Gamma[np.mod(i_bond+1+i,2)],axes=(0,1))
                    T=np.tensordot(T,np.conj(self.Gamma[np.mod(i_bond+1+i,2)]),axes=(0,1))
                    R=np.trace(T,axis1=0,axis2=2)
                C=np.tensordot(self.Gamma[np.mod(i_bond+dist,2)],np.conj(self.Gamma[np.mod(i_bond+dist,2)]),axes=(2,2))
                L=np.tensordot(R,C,axes=([0,1],[1,3]))
                if connected:
                   corr += np.real(np.tensordot(L,sz,axes=([0,1],[0,1])) - mean1*mean1)
                else:
                   corr += np.real(np.tensordot(L,sz,axes=([0,1],[0,1]))) 

            return 0.5*corr


    def spectrum(self):
        return self.Lambda[0]*self.Lambda[0].conj()
       
