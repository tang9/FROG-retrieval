"""CO2 disperison data (in terms of wavelength in um) from 
J. Opt. Soc. Am. 61, 89-90 (1971)
"""

import numpy as np
from numpy.lib.scimath import sqrt
Pi=np.pi

lambda_min=0.48
lambda_max=1.82

def n(x):  
    return 1+0.00000154489/(0.0584738-x**-2)+0.083091927/(210.9241-x**-2)+0.0028764190/(60.122959-x**-2)


# """nonlinear dispersion"""

# def chi2():
#     """returns chi2(:,:,:)
#     might be wrong I didnot check"""
    
#     chi=np.zeros((3,3,3))

#     return 2*chi #2 comes from the historic definition of d

# def chi3():
#     """returns chi3(:,:,:,:)
#     https://doi.org/10.1007/10134958_36
#     """
#     c_xxxx=1.6*10**-22 #m^2/V^2. approximately. might be 2x less
#     c_xxyy=0.6*10**-22 #m^2/V^2
    
#     chi=np.zeros((3,3,3,3))
    
#     #since it is a cubic crystal
#     chi[0][0][0][0]=c_xxxx
#     chi[1][1][1][1]=c_xxxx
#     chi[2][2][2][2]=c_xxxx
    
#     chi[0][0][1][1]=c_xxyy
#     chi[0][1][0][1]=c_xxyy
#     chi[0][1][1][0]=c_xxyy
#     chi[0][0][2][2]=c_xxyy
#     chi[0][2][0][2]=c_xxyy
#     chi[0][2][2][0]=c_xxyy
#     chi[1][1][0][0]=c_xxyy
#     chi[1][0][1][0]=c_xxyy
#     chi[1][0][0][1]=c_xxyy
#     chi[1][1][2][2]=c_xxyy
#     chi[1][2][1][2]=c_xxyy
#     chi[1][2][2][1]=c_xxyy
#     chi[2][2][0][0]=c_xxyy
#     chi[2][0][2][0]=c_xxyy
#     chi[2][0][0][2]=c_xxyy
#     chi[2][2][1][1]=c_xxyy
#     chi[2][1][2][1]=c_xxyy
#     chi[2][1][1][2]=c_xxyy
    
#     return chi