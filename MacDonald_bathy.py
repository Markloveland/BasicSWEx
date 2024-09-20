#solves for bathymetry in MacDonalds
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, RK45


#exact solution
def h_exact(x,g=9.81):
	return ((4.0/g)**(1/3)*(1-0.2*np.exp(-36*(x/1000 - 0.5)**2)) )

#derivative of exact solution
def dhdx_exact(x,g=9.81):
    return ((4.0/g)**(1/3)*((0.0000144*x - .0072)*np.exp(-9*(x-500)**2/250000)) )  

def get_quadratic_Sf(q,cf,h,g=9.81):
    return (cf/(8*g))*q*q/(h**(3))

def get_mannings_Sf(q,n,h,g=9.81):
    return n*n*q*q/(h**(10/3)) 
#try automatic differentiation
x0=0
x1=1000
nx = 101
x=np.linspace(x0,x1,nx)
h_x = h_exact(x)
plt.plot(x,h_x)
plt.savefig("h_exact.png")
plt.close()
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
    Sf = (cf/(8*g))*q*q/(h**(3))
    return np.array([(q**2/(g*h**3) - 1)*dxdh - Sf])

plt.plot(x,get_bath_grad(x,0).flatten())

dydt = get_bath_grad(x,0)

y0 = 0           # initial value y0=y(t0)
#sol = solve_ivp(fun=get_bath_grad, t_span=[x0, x1], y0=[y0],atol=1e-8, rtol=1e-8)  # computation of SOLution 

integrator = RK45(get_bath_grad, 0.0, np.array([0.0]), x1,max_step=0.1)
t_values = []
y_values = []

while integrator.t < x1:
    integrator.step()
    t_values.append(integrator.t)
    y_values.append(integrator.y)

t_values = np.array(t_values)
y_values = np.array(y_values).flatten()


plt.plot(t_values,np.gradient(y_values,t_values),'--')
plt.savefig("Bathygrad.png")
plt.close()

plt.plot(t_values,y_values)

#plt.plot(sol.t,sol.y[0])
plt.savefig("Bathy.png")