import numpy as np
class mps:


    def __init__(self,site_dimension=2,bond_dimension=100):
        self.site_dimension = site_dimension
        self.bond_dimension = bond_dimension
        self.Gamma=[]
        self.Lambda=[]
#        self.product_state()




    @property
    def site_dimension(self):
        print("getting single site hilbert space dimension")
        return self._site_dimension

    @property
    def bond_dimension(self):
        print("getting bond dimension")
        return self._bond_dimension

    @property
    def Gamma(self):
        return self._Gamma

    @property
    def Lambda(self):
        return self._Lambda

    @site_dimension.setter
    def site_dimension(self, d):
        print("setting single site hilbert space dimension")
        self._site_dimension=d

    @bond_dimension.setter
    def bond_dimension(self, chi):
        print("setting single site hilbert space dimension")
        self._bond_dimension=chi

    @Gamma.setter
    def Gamma(self,gamma):
        self._Gamma=gamma

    @Lambda.setter
    def Lambda(self,lambd):
        self._Lambda=lambd

    def product_state(self):
        d=self.site_dimension
        for i in range(2):
            self.Lambda.append(np.ones([1]))
            self.Gamma.append(np.zeros([d,1,1]))
            self.Gamma[-1][0,0,0]=1

