from pathlib import Path
import numpy as np
from dolfinx import fem as fe, mesh,io
from dolfinx import __version__
from mpi4py import MPI
from ufl import (
    TestFunction, TrialFunction, FacetNormal, as_matrix,
    as_vector, as_tensor, dot, inner, grad, dx, ds, dS,
    jump, avg,sqrt,conditional,gt,div,nabla_div,tr,diag,sign,elem_mult,
    TestFunctions, Measure
)

try:
  from ufl import FiniteElement, VectorElement, MixedElement
  use_basix=False
except ImportError:
  use_basix=True
  from basix.ufl import element, mixed_element
try:
  from dolfinx.fem import functionspace
except ImportError:
  from dolfinx.fem import FunctionSpace as functionspace

from petsc4py.PETSc import ScalarType
from boundaryconditions import BoundaryCondition,MarkBoundary
from newton import CustomNewtonProblem
from auxillaries import init_stations, record_stations, gather_stations
import matplotlib.pyplot as plt
#store version as 3 integers
release_no = int(__version__[0])
version_no = int(__version__[2])
revision_no = int(__version__[4])
#plotting functionality depends on version
if release_no<=0 and version_no<8:
	use_vtx=False
else:
	use_vtx=True
#######################################################################
#General user inputs here#
#Filename for where outputs will go
filename='TidalPropagation'
#global output for every "plot_every" time steps
plot_every=1
#any user defined solver paramters
rel_tol=1e-5
abs_tol=1e-6
max_iter=10
relax_param=1
params = {"rtol": rel_tol, "atol": abs_tol, "max_it":max_iter, "relaxation_parameter":relax_param, "ksp_type": "gmres", "pc_type": "bjacobi"}
#Provide any points where you would like to record time series data

wd_alpha = 0.5
########################################################################
########################################################################
#######Define the physical domain########
#For this basic script, just a rectangle

#First define physical dimensions
#Coordinate of bottom left corner
x0 = 0.0
y0 = 0.0
#Coordinate of top right corner
x1= 600.0
y1= 600.0


#Now define mesh properties
#number of cells in x and y direction
nx=100
ny=100

#For n stations the np array should be nx3
stations = np.array([[x1/2,y1/2,0.0]])

#creates dolfinx mesh object partioned via MPI
domain = mesh.create_rectangle(MPI.COMM_WORLD, [[x0, y0],[x1, y1]], [nx, ny])

####################################################################################
###################################################################################
##########Define Length of simulation and time step size##########################
#ts is start time in seconds
ts=0.0
#tf is final time in seconds
tf=8#7*24*60*60
#time step size in seconds
dt=0.1
#####################################################################################
####We need to identify function spaces before we can assign initial conditions######

#We will use "DG" elements
p_type = "DG"
#polynomial order of finite element, h is first , (u,v) is second
p_degree = [1,1]

# We use mixed elements, these are ufl objects for symbolic math
if not use_basix:
	#<080
	el_h   = FiniteElement(p_type, domain.ufl_cell(), degree=p_degree[0])
	el_vel = VectorElement(p_type, domain.ufl_cell(), degree=p_degree[1], dim = 2)
	#V will hold the function space info of the mixed elements
	V = functionspace(domain, MixedElement([el_h, el_vel]))
else:
	#>=080
	el_h   = element(p_type, domain.basix_cell(), degree=p_degree[0])
	el_vel = element(p_type, domain.basix_cell(), degree=p_degree[1], shape = (2,))
	V = functionspace(domain, mixed_element([el_h, el_vel]))

V_scalar = V.sub(0).collapse()[0]
V_vel = V.sub(1).collapse()[0]
#solution variables
#this will store solution as we march through time
u = fe.Function(V)
print("Length of solution = ",u.x.array.shape)
#split into h, and velocity
h, vel = u.split()

#also solutions for previous time steps
u_n = fe.Function(V)
u_n_old = fe.Function(V)
#function that stores any dirichlet boundary conditions
u_ex = fe.Function(V)

#Create test functions
p1, p2 = TestFunctions(V)
# object that concatenates all test functions into single variable, like u
p = as_vector((p1,p2[0],p2[1]))

################################################################################
################################################################################

####Assigning bathymetry and initial conditions#########
##Note that fenicsx can handle bathymetry from different function space
#But we will keep same for now

#Bathymetry assignment

#for this problem, assume uniform depth of 10 m
depth=10.0
slope = 2*depth/x1
DG0 = fe.functionspace(domain, ("DG", 0))
h_b = fe.Function(V_scalar)
#Event though constant still needs to be function of x by convention
h_b.interpolate(lambda x: 0*x[0])

h_init = fe.Function(DG0)
h_init.interpolate(lambda x: depth*(x[0]< (x1/2)))


#Initial condition assignment
#in this case, initial condition is h=h_b, vel=(0,0)
u_n.sub(0).interpolate(
	fe.Expression(
		h_init, 
		V.sub(0).element.interpolation_points()))
  
u_n.sub(1).interpolate(
	fe.Expression(
		as_vector([fe.Constant(domain, ScalarType(0.0)),
			fe.Constant(domain, ScalarType(0.0))]),
		V.sub(1).element.interpolation_points()))

#also need to input bathymetry to u_ex to store h_b
h_ex = u_ex.sub(0)
h_ex.interpolate(
	fe.Expression(
		h_init,
		V.sub(0).element.interpolation_points())
	)

################################################################################
################################################################################

#####Time dependent boundary conditions######
#####Define boundary condition locations#####

# 1 is the left side, and will be a tidal boundary condition (Dirichlet condition for surface elevation)
# 2 are all other sides and are no flux condition (U \cdot n = 0)
# We can add more numbers for different bc types later
boundaries = [(1, lambda x: np.isclose(x[0], 0)),
              (2, lambda x: np.logical_not(np.isclose(x[0],0 )) | np.isclose(x[1],y1) |  np.isclose(x[1],y0))]


##########Defining functions which actually apply the boundary conditions######
facet_markers, facet_tag = MarkBoundary(domain, boundaries)
#generate a measure with the marked boundaries
ds = Measure("ds", domain=domain, subdomain_data=facet_tag)


##########Dirchlet Boundary conditions###################
# Define the boundary conditions and pass them to the solver
h_dirichlet_conditions = []
#For now these will be empty but in general may want to allow for dirichlet u,v
ux_dirichlet_conditions = []
ux_dirichlet_dofs = np.array([])
uy_dirichlet_conditions = []
uy_dirichlet_dofs = np.array([])

#identify equation numbers associated with boundary conditions
#this is only necessary for dirichlet conditions
#can add in more later
#the dirichlet_conditions are a list containing dolfinx functions that assign bc
for marker, func in boundaries:
	if marker == 1:
		h_dirichlet_dofs,bc = BoundaryCondition("Open", marker, func, u_ex.sub(0), V.sub(0))
		h_dirichlet_conditions.append(bc)



###To prepare for time loop, assign the boundary forcing function, u_ex
#this will compute the tidal elevation at the boundary
def evaluate_tidal_boundary(t):
	#hard coded parameters for mag and frequency
	alpha = 0.00014051891708
	mag = 0.0#0.15
	return mag*np.cos(t*alpha)


#A function which will update the dirichlet bc inside the time loop
def update_boundary(t,hb):
	#take in time and return float of tide level
	tide_level = evaluate_tidal_boundary(t)
	#return np array with tide level at dirichlet boundary
	return hb + tide_level



#define h_b at boundary of dof so we don't need to repeat in time loop
#hb_boundary is a vector that will not change through time
hb_boundary = h_ex.x.array[h_dirichlet_dofs]

################################################################################
################################################################################

#######Establish weak form to solve within time loop########


################################################################################
###############################################################################
#add in wd_f from Karna
#this will be added to h_b
def wd_f(alpha,h):
	return 0.5*(sqrt(h**2 + alpha*alpha) - h)




#aliasing to make reading easier
h, ux, uy = u[0], u[1], u[2]
#get wd function, this will be added anywhere involving h, h_b
h_wd = h + wd_f(wd_alpha,h)
h_n_wd = u_n[0] + wd_f(wd_alpha,u_n[0])
h_n_old_wd =u_n_old[0] + wd_f(wd_alpha,u_n_old[0])
h_b_wd = h_b + wd_f(wd_alpha,h)

#also shorthand reference for flux variable:
#this is the modified form now
Q      =   as_vector((h_wd, u[1]*h_wd, u[2]*h_wd ))
Qn     =   as_vector((h_n_wd, u_n[1]*h_n_wd, u_n[2]*h_n_wd))
Qn_old =   as_vector((h_n_old_wd, u_n_old[1]*h_n_old_wd, u_n_old[2]*h_n_old_wd )) 


#g is gravitational constant
g=9.81


#Flux tensor from SWE
Fu = as_tensor([[h_wd*ux,h_wd*uy], 
				[h_wd*ux*ux + 0.5*g*h_wd*h_wd, h_wd*ux*uy],
				[h_wd*ux*uy,h_wd*uy*uy+0.5*g*h_wd*h_wd]
				])

#Flux tensor for SWE if normal flow is 0
Fu_wall = as_tensor([[0,0], 
					[0.5*g*h_wd*h_wd, 0],
					[0,0.5*g*h_wd*h_wd]
					])


#RHS source vector for SWE is gravity + bottom friction
#can add in things like wind and pressure later
g_vec = as_vector((0,
 					-g*h_wd*h_b_wd.dx(0),
 					-g*h_wd*h_b_wd.dx(1)))

#bf
eps=1e-8
cf=0.00025
mag_v = conditional(pow(ux*ux + uy*uy, 0.5) < eps, 0, pow(ux*ux + uy*uy, 0.5))
fric_vec=as_vector((0,
                    ux*cf*mag_v,
                    uy*cf*mag_v))

S = g_vec#+fric_vec



#normal vector
n = FacetNormal(domain)




#begin constructing the weak form, this is standard weak form from IBP
#start adding to residual, beggining with body term
F = -inner(Fu,grad(p))*dx
#add RHS forcing
F += inner(S,p)*dx


#now adding in global boundary terms
for marker, func in boundaries:
	if marker == 1:
		#This is the open boundary in this case
		F += dot(dot(Fu, n), p) * ds(marker)
	elif marker == 2:
		#this is the wall condition, no flux on this part
		F += dot(dot(Fu_wall, n), p)*ds(marker)

#now adding interior boundary terms using Lax-Friedrichs upwinding for DG
eps=1e-8
#attempt at full expression from https://docu.ngsolve.org/v6.2.1810/i-tutorials/unit-3.4-simplehyp/shallow2D.html
vela =  as_vector((u[1]('+'),u[2]('+')))
velb =  as_vector((u[1]('-'),u[2]('-')))
vnorma = conditional(sqrt(dot(vela,vela)) > eps,sqrt(dot(vela,vela)),0)
vnormb = conditional(sqrt(dot(velb,velb)) > eps,sqrt(dot(velb,velb)),0)
C = conditional( (vnorma + sqrt(g*h_wd('+'))) > (vnormb + sqrt(g*h_wd('-'))), (vnorma + sqrt(g*h_wd('+'))) ,  (vnormb + sqrt(g*h_wd('-')))) 
flux = dot(avg(Fu), n('+')) + 0.5*C*jump(Q)

F += inner(flux, jump(p))*dS


#now add terms related to time step
#specifies time stepping scheme, save it as fe.constant so it is modifiable
theta=1
theta1 = fe.Constant(domain, ScalarType(theta))


# this is a generalized version of the BDF2 scheme
#theta1=0 is 1st order implicit Euler, theta1=1 is 2nd order BDF2
dQdt = theta1*fe.Constant(domain,ScalarType(1.0/dt))*(1.5*Q - 2*Qn + 0.5*Qn_old) + (1-theta1)*fe.Constant(domain,ScalarType(1.0/dt))*(Q - Qn)
#add to weak form
F+=inner(dQdt,p)*dx


#Weak form and initial conditions are now arranged, 
# Almost ready for Main time loop but need to initialize a few thing

###################################################################################
###################################################################################
#Some final steps before the main time loop
#Uses FEniCSx to advance time by using Newton-Raphson for each implicit time step

#First let's create plots so we can view initial condition before advancing in time
#initiate some auxillary functions for plotting
#it is more common to plot surface elevation (eta) rather than depth (h)        
eta_plot = fe.Function(V_scalar)
eta_plot.name = "WSE(m)"

h_plot = fe.Function(V_scalar)
h_plot.name = "depth(m)"

vel_plot = fe.Function(V_vel)
vel_plot.name = "depth averaged velocity (m/s)"

#060
if not use_vtx:
	xmf = io.XDMFFile(domain.comm, filename+"/"+filename+".xdmf", "w")
	xmf.write_mesh(domain)
	writers=[]
else:
	#080
	xmf=None
	results_folder = Path(filename)
	results_folder.mkdir(exist_ok=True, parents=True)
	wse_writer = io.VTXWriter(domain.comm, results_folder / "WSE.bp", eta_plot, engine="BP4")
	h_writer = io.VTXWriter(domain.comm, results_folder / "h.bp", h_plot, engine="BP4")
	vel_writer = io.VTXWriter(domain.comm, results_folder / "vel.bp", vel_plot, engine="BP4")
	writers=[wse_writer,h_writer,vel_writer]

def plot_global_output(u,h_b,V_scalar,V_vel,t,xdmf=None,vtx_writers=None):
	#interpolate and plot water surface elevation and velocity
	h_expr = fe.Expression(u.sub(0).collapse(), V_scalar.element.interpolation_points())
	h_plot.interpolate(h_expr)
	eta_expr = fe.Expression(u.sub(0).collapse() - h_b, V_scalar.element.interpolation_points())
	eta_plot.interpolate(eta_expr)
	v_expr = fe.Expression(u.sub(1).collapse(), V_vel.element.interpolation_points())
	vel_plot.interpolate(v_expr)
	
	#060
	if xdmf!=None:
		xdmf.write_function(eta_plot,t)
		xdmf.write_function(vel_plot,t)
	else:
		#080
		for a in vtx_writers:
			a.write(t)
	return 0



#######Initialize a solver object###########
#utilize the custom Newton solver class instead of the fe.petsc Nonlinear class
Newton_Solver = CustomNewtonProblem(F,u,h_dirichlet_conditions, domain.comm, solver_parameters=params)



#################################################################################
#################################################################################

#begin the main loop now
nt=int(np.ceil((tf-ts)/dt))
#set up array to record any time series at points
local_cells,local_points = init_stations(domain,stations)
station_data = np.zeros((nt+1,local_points.shape[0],3))
#record initial data
station_data[0,:,:] = record_stations(u_n,local_points,local_cells)

#time begins at ts
t=ts

#take first 2 steps with implicit Euler since we dont have enough steps for higher order
theta1.value=0
u.x.array[:] = u_n.x.array[:]

#plot initial condition
plot_global_output(u,h_b,V_scalar,V_vel,ts,xdmf=xmf,vtx_writers=writers)
for a in range(min(2,nt)):
	#by default we print out on screen each time step
	print('Time Step Number',a,'Out of',nt)
	print(a/nt*100,'% Complete')
	print(len(u.x.array[:]))
	#save new solution as previous solution
	u_n_old.x.array[:] = u_n.x.array[:]
	u_n.x.array[:] = u.x.array[:]
	#update time
	t += dt
	#update any dirichlet boundary conditions
	#for now just the one but may expand in future
	u_ex.sub(0).x.array[h_dirichlet_dofs] = update_boundary(t,hb_boundary)
	u.x.array[h_dirichlet_dofs] = u_ex.x.array[h_dirichlet_dofs]
	#solve associated NewtonProblem
	Newton_Solver.solve(u)
	#add data to station variable
	station_data[a+1,:,:] = record_stations(u,local_points,local_cells)
	#Plot global solution
	if a%plot_every==0 and plot_every <= nt:
		plot_global_output(u,h_b,V_scalar,V_vel,t,xdmf=xmf,vtx_writers=writers)

#Take remainder of time steps with 2nd order BDF2 scheme
theta1.value=1
for a in range(2, nt):
	print('Time Step Number',a,'Out of',nt)
	print(a/nt*100,'% Complete')
	u_n_old.x.array[:] = u_n.x.array[:]
	u_n.x.array[:] = u.x.array[:]
	#update time
	t += dt
	#update any dirichlet boundary conditions
	#for now just the one but may expand in future
	u_ex.sub(0).x.array[h_dirichlet_dofs] = update_boundary(t,hb_boundary)
	u.x.array[h_dirichlet_dofs] = u_ex.x.array[h_dirichlet_dofs]
	#solve associated NewtonProblem
	Newton_Solver.solve(u)
	#add data to station variable
	station_data[a+1,:,:] = record_stations(u,local_points,local_cells)
	#Plot global solution
	if a%plot_every==0:
		plot_global_output(u,h_b,V_scalar,V_vel,t,xdmf=xmf,vtx_writers=writers)


#################################################################################
#################################################################################

#Time loop is complete, any postprocessing may go here
#060
if not use_vtx:
	xmf.close()
else:
	#080
	for writer in writers:
		writer.close()

#These variables will hold the coordinates and corresponding values of all stations
coords,vals = gather_stations(0,domain.comm,local_points,station_data)

#example of post processing
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank ==0:
	t_vec=np.linspace(ts,tf,nt+1)
	sec2day=60*60*24
	t_vec=t_vec/sec2day
	#optionally save csv or plot line plots of stations
	#save array for post processing
	np.savetxt(f"{filename}_stations_h.csv", vals[:,:,0], delimiter=",")
	np.savetxt(f"{filename}_stations_xvel.csv", vals[:,:,1], delimiter=",")
	np.savetxt(f"{filename}_stations_yvel.csv", vals[:,:,2], delimiter=",")
	plt.plot(t_vec, vals[:,:,0].flatten(), "--", linewidth=2, label="h at x= "+str(stations[0,0]))
	plt.grid(True)
	plt.xlabel("t(days)")
	plt.ylabel('surface elevation(m)')
	plt.title(f'Surface Elevation over Time')
	plt.legend()
	plt.savefig(f"{filename}_h_station.png")
	plt.close()
	plt.plot(t_vec, vals[:,:,1].flatten(), "--", linewidth=2, label="ux at "+str(stations[0,0]))
	plt.grid(True)
	plt.xlabel("t(days)")
	plt.title(f'Velocity in x-direction Over Time')
	plt.legend()
	plt.savefig(f"{filename}_ux_station.png")

