"""GaSe disperison data (in terms of wavelength in um) from 
Appl. Opt. 52, 2325-2328 (2013)"""

import os
import sys
Path=os.path.dirname((os.path.abspath(__file__)))
SP=Path.split("\\")
i=0
while i<len(SP) and SP[i].find('python')<0:
    i+=1
Pypath='\\'.join(SP[:i+1])
sys.path.append(Pypath)
# speed of light in vacuum (m/s)
c = 299792458.0

# vacuum permittivity (F/m)
e0 = 8.854187817e-12
import numpy as np
from numpy.lib.scimath import sqrt
from numpy.linalg import inv
Pi=np.pi

lambda_min=0.8
lambda_max=162

def n(x):
    # no=sqrt(10.6409+0.3788/(x**2-0.1232)+7090.7/(x**2-2216.3))
    # ne=sqrt(8.2477+0.2881/(x**2-0.1669)+4927.5/(x**2-1990.1))
    no=(10.6409+0.3788/(x**2-0.1232)+7090.7/(x**2-2216.3))**0.5
    ne=(8.2477+0.2881/(x**2-0.1669)+4927.5/(x**2-1990.1))**0.5
    return np.array([no,ne])
    # return no

"""nonlinear dispersion"""
def chi2():
    """returns chi2(:,:,:)
    SNLO JETP Lett. v16 p90 (1972) 54"""
    d22=58*10**-12 #m/V  Eksma: 63?

    chi=np.zeros((3,3,3))
    chi[1][0][0]=-d22

    chi[1][1][1]=d22
    
    chi[0][0][1]=chi[0][1][0]=-d22

    return 2*chi #2 comes from the historic definition of d

def chi3(lam=1):
    """returns chi3(:,:,:,:)
    """
    n2=450*10**-20 #m^2/W nonlinear refrective index . 
    #Laser Photon. Rev. 2, 11 (2008)
    
    #it is not really correct because there has to be some dependence for a crystal but I have not found it
    c_xxxx=n2*4/3*c*e0*n(lam)[0]**2 #m^2/V^2
    
    chi=np.zeros((3,3,3,3))
    
    chi[0][0][0][0]=c_xxxx
    chi[1][1][1][1]=c_xxxx
    chi[2][2][2][2]=c_xxxx
    
    return chi