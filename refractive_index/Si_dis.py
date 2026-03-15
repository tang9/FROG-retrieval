"""Si disperison data (in terms of wavelength in um)
"""

import numpy as np
# speed of light in vacuum (m/s)
c = 299792458.0

# vacuum permittivity (F/m)
e0 = 8.854187817e-12
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import os
Path=os.path.dirname((os.path.abspath(__file__)))
from scipy.optimize import curve_fit

# interpolated=True

file=Path+'\\Si\\Si_1.4.txt' #AIP Advances 5, 67168 (2015)
Si_1=np.loadtxt(file,skiprows=1)
InterpolSi1=interpolate.PchipInterpolator(Si_1[:,0],Si_1[:,1])
Si_1_lambda_min=Si_1[0][0]
Si_1_lambda_max=Si_1[-1][0]

file=Path+'\\Si\\Si_2.txt' #J. Phys. Chem. Ref. Data 9, 561-658 (1993)
Si_2=np.loadtxt(file,skiprows=1)
InterpolSi2=interpolate.PchipInterpolator(Si_2[:,0],Si_2[:,1])
Si_2_lambda_min=Si_2[0][0]
Si_2_lambda_max=Si_2[-1][0]

lambda_min20=2.5
lambda_min=Si_1_lambda_min
lambda_max=22.2

# def n20(x):  
#     """J. Appl. Phys. 97, 123526 (2005)"""
#     # return 3.41983+0.159906/(x**2-0.028)-0.123109/(x**2-0.028)**2+1.26878E-6*x**2-1.95104E-9*x**4
#     return (11.67316+1/(x**2)+0.004482633/(x**2-1.108205**2))**.5

# #correction coefficienct to avoid jumps at joins
# cor2=InterpolSi1(Si_1_lambda_max)-InterpolSi2(Si_1_lambda_max)
# cor20=cor2+InterpolSi2(lambda_min20)-n20(lambda_min20)

# def n_com(x):
#     """combined refractive index"""
#     # return InterpolSi1(x)
#     # print(x,x <= Si_1_lambda_max)
#     if x <= Si_1_lambda_max:
#         return InterpolSi1(x)
#     elif Si_1_lambda_max<x<lambda_min20:
#         return InterpolSi2(x) + cor2
#     else:
#         return n20(x)+cor20


# def fitF(x,a,b,c,d,f):
#         return (a+b/(x**2-c)+d/(x**2-f))**0.5

# X=np.linspace(Si_1_lambda_min,lambda_max,100)
# Y=np.vectorize(n_com)(X)
# popt, _ = curve_fit(fitF, X, Y, p0=[10,1,0,0.00448,1.108205**2],
#                     bounds=((-np.inf,-np.inf,-np.inf,-np.inf,Si_1_lambda_min**2),
#                             (np.inf,np.inf,Si_1_lambda_min**2,np.inf,np.inf)))

# print(popt)

# def n(x):
#     """new interpolated function"""
#     print(fitF(x,*popt))
#     return fitF(x,*popt)

def n(x):  
    # """J. Appl. Phys. 97, 123526 (2005)"""
    # return 3.41983+0.159906/(x**2-0.028)-0.123109/(x**2-0.028)**2+1.26878E-6*x**2-1.95104E-9*x**4
    # return (11.67316+1/(x**2)+0.004482633/(x**2-1.108205**2))**.5
    return (11.67335772+0.2163348/(x**2-0.01290199)+0.75460663/(x**2-0.11900219))**0.5

# print(popt)
# plt.clf()
# W=np.linspace(0.2,3,100)
# plt.plot(W,np.vectorize(n)(W))
# plt.plot(W,np.vectorize(n_com)(W))
# plt.show()


"""nonlinear dispersion"""

def chi2():
    """returns chi2(:,:,:)
    might be wrong I didnot check"""
    
    chi=np.zeros((3,3,3))

    return 2*chi #2 comes from the historic definition of d

def chi3(lam=3.5):
    """returns chi3(:,:,:,:)
    """
    n2=3.6*10**-18 #m^2/W nonlinear refrective index.@3.5um Nanophotonics 2014; 3(4-5) 247–268

    c_xxxx=n2*4/3*c*e0*n(lam)**2 #m^2/V^2
    
    chi=np.zeros((3,3,3,3))
    
    chi[0][0][0][0]=c_xxxx
    chi[1][1][1][1]=c_xxxx
    chi[2][2][2][2]=c_xxxx
    
    return chi