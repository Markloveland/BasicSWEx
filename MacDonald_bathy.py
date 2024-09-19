#solves for bathymetry in MacDonalds
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


#exact solution
def h_exact(x,g=9.81):
	return ((4.0/g)**(1/3)*(1-0.2*np.exp(-36*(x/1000 - 1/2)**2)) )

#derivative of exact solution
def dhdx_exact(x,g=9.81):
    return ((4.0/g)**(1/3)*((0.0000144*x - .0072)*np.exp(-9*(x-500)**2/250000)) )  

def get_quadratic_Sf(q,cf,h,g=9.81):
    return g*(cf/8*g)*q*q/(h**(2))

def get_mannings_Sf(q,n,h,g=9.81):
    return g*n*n*q*q/(h**(7/3)) 
#try automatic differentiation
x0=0
x1=1000
nx = 101
x=np.linspace(x0,x1,nx)
h_x = h_exact(x)
dhdx_x = dhdx_exact(x)
dhdx_cd = np.gradient(h_x,x[1]-x[0])
print(np.sum((dhdx_x-dhdx_cd)**2))



#now get bathy from friction
cf = .065
q= 2.5
Sf = get_quadratic_Sf(q,cf,h_x)

def get_bath_grad(x,z):
    g=9.81
    h = (4.0/g)**(1/3)*(1-0.2*np.exp(-36*(x/1000 - 1/2)**2))
    dxdh = (4.0/g)**(1/3)*((0.0000144*x - .0072)*np.exp(-9*(x-500)**2/250000))
    Sf = g*(cf/8*g)*q*q/(h**(2))
    return (q**2/(g*h**3) - 1)*dxdh - Sf

plt.plot(x,get_bath_grad(x,0))
plt.savefig("Bathygrad.png")
plt.close()

y0 = 0           # initial value y0=y(t0)
sol = solve_ivp(fun=get_bath_grad, t_span=[x0, x1], y0=[y0],atol=1e-8, rtol=1e-8)  # computation of SOLution 

plt.plot(sol.t,sol.y[0])
plt.savefig("Bathy.png")