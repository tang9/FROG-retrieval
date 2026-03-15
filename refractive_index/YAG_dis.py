"""Y3Al5O12 (Yttrium aluminium garnet, YAG) disperison data (in terms of wavelength in um) from 
, Appl. Opt. 37, 4933-4935 (1998)
0.4-5 µm
"""

from numpy.lib.scimath import sqrt
import numpy as np
# speed of light in vacuum (m/s)
c = 299792458.0

# vacuum permittivity (F/m)
e0 = 8.854187817e-12

lambda_min=0.4
lambda_max=5

def n(x):  
    # return sqrt(1+2.28200/(1-0.01185/x**2)+3.27644/(1-282.734/x**2))
    return (1+2.28200/(1-0.01185/x**2)+3.27644/(1-282.734/x**2))**0.5


"""nonlinear dispersion"""

def chi2():
    """returns chi2(:,:,:)
    might be wrong I didnot check"""
    
    chi=np.zeros((3,3,3))

    return 2*chi #2 comes from the historic definition of d

def chi3(lam=1):
    """returns chi3(:,:,:,:)
    """
    n2=6.5*10**-20 #m^2/W nonlinear refrective index. Opt. Express 27, 11018-11028 (2019). 8*10**-20 AIP Conference Proceedings 1665, 060010 (2015)

    #it is not really correct because there has to be some dependence for a crystal but I have not found it
    c_xxxx=n2*4/3*c*e0*n(lam)**2 #m^2/V^2
    
    chi=np.zeros((3,3,3,3))
    
    chi[0][0][0][0]=c_xxxx
    chi[1][1][1][1]=c_xxxx
    chi[2][2][2][2]=c_xxxx
    
    #assuming isotropic medium
    c_xxyy=c_xxxx/3 #m^2/V^2
    
    chi[0][0][1][1]=c_xxyy
    chi[0][1][0][1]=c_xxyy
    chi[0][1][1][0]=c_xxyy
    chi[0][0][2][2]=c_xxyy
    chi[0][2][0][2]=c_xxyy
    chi[0][2][2][0]=c_xxyy
    chi[1][1][0][0]=c_xxyy
    chi[1][0][1][0]=c_xxyy
    chi[1][0][0][1]=c_xxyy
    chi[1][1][2][2]=c_xxyy
    chi[1][2][1][2]=c_xxyy
    chi[1][2][2][1]=c_xxyy
    chi[2][2][0][0]=c_xxyy
    chi[2][0][2][0]=c_xxyy
    chi[2][0][0][2]=c_xxyy
    chi[2][2][1][1]=c_xxyy
    chi[2][1][2][1]=c_xxyy
    chi[2][1][1][2]=c_xxyy
    
    return chi