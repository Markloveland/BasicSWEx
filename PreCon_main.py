import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
from dolfinx import fem as fe, mesh,io
#from dolfinx.graph import adjacencylist
from mpi4py import MPI
import time
'''
from ufl import (
    VectorElement, TestFunction, TrialFunction, FacetNormal, as_matrix,
    as_vector, as_tensor, dot, inner, grad, dx, ds, dS,
    jump, avg, sqrt,conditional,gt,div,nabla_div,tr,diag,sign,elem_mult,
    MixedElement, FiniteElement, TestFunctions, Measure, Mesh, VectorElement, Cell
)
'''
import ufl
from petsc4py.PETSc import ScalarType
from boundaryconditions import BoundaryCondition,MarkBoundary
from newton import CustomNewtonProblem
from newton2 import CustomNewtonProblem2
from auxillaries import init_stations, record_stations, gather_stations
import matplotlib.pyplot as plt
#######################################################################
#General user inputs here#
#Filename for where outputs will go
filename='AdvectionTest'
#global output for every "plot_every" time steps
plot_every=1
#any user defined solver paramters
rel_tol=1e-5
abs_tol=1e-6
max_iter=10
relax_param=1
params = {"rtol": rel_tol, "atol": abs_tol, "max_it":max_iter, "relaxation_parameter":relax_param, "ksp_type": "gmres", "pc_type": "line_smooth"}
#Provide any points where you would like to record time series data
#For n stations the np array should be nx3
stations = np.array([[1000.5,1000.5,0.0]])
########################################################################
########################################################################
#######Define the physical domain########
#For this basic script, just a rectangle

#First define physical dimensions
#Coordinate of bottom left corner
x0 = 0.0
y0 = 0.0
#Coordinate of top right corner
y1= 2000.0
x1= 2000.0


#Now define mesh properties
#number of cells in x and y direction
nx=20#20
ny=20#5

#Propagation velocity entering the left side
#A proper bc should be flux maybe?
depth = 2
vel_boundary_mag = 9.0
flux_boundary_mag = depth*vel_boundary_mag

#creates dolfinx mesh object partioned via MPI
#domain = mesh.create_rectangle(MPI.COMM_WORLD, [[x0, y0],[x1, y1]], [nx, ny],mesh.CellType.quadrilateral)

#Alternatively, build a mesh and force elements to be same order as streamlines
#just for proof of concept

#given nx by ny, start at top of mesh and go left to right
x_coords = np.linspace(x0,x1,nx+1)
y_coords = np.linspace(y0,y1,ny+1)
#now tile them to create inufl.dividual nodes
x_coords = np.tile(x_coords,ny+1)
y_coords = np.repeat(y_coords,nx+1)
coords=np.column_stack([x_coords,y_coords])

#create cell order so that it goes left to right
nm = np.zeros((nx*ny*2,3),dtype=np.int32)
cell_no =0
idx=0
for a in range(ny):
	for b in range(nx):
		nw_node =  idx+(nx+1)
		ne_node = idx+(nx+2)
		sw_node = idx
		se_node = idx+1
		nm[cell_no] = np.array([sw_node,ne_node,nw_node])
		cell_no+=1
		nm[cell_no] = np.array([sw_node,se_node,ne_node])
		cell_no+=1
		idx+=1
		if b==nx-1:
			#skip one at end of column
			idx+=1
#print(coords.shape)
print("Orignal nodes and connectivity")
print(coords)
#print(nm.shape)
print(nm)



gdim, shape, degree = 2, "triangle", 1
cell = ufl.Cell(shape, geometric_dimension=gdim)
element = ufl.VectorElement("Lagrange", cell, degree)
domain = mesh.create_mesh(MPI.COMM_SELF, nm, coords, ufl.Mesh(element))



#our original streamline direction for left to right flow is 2*the number of elements in x
#original lines
lines = np.arange(nx*ny*2).reshape(nx*2,ny,order='F')


#need a function that finds map from new element to original element number
#we know the node mapping through
#domain.geometry.input_global_indices
#we need to find matching elements
#print(domain.geometry.input_global_indices)
#what happens here

new_2_old = np.argsort(domain.geometry.input_global_indices)
old_2_new = domain.geometry.input_global_indices

#these are the mappings
#this gives the old nodes
print("new nodes and new connectivity")
print(domain.geometry.x[:])
print(domain.geometry.dofmap)
#print("old nodes mapped to new")
#print(coords[old_2_new,:])



def find_element_mapping(domain,coord,nm):
	ncell = nm.shape[0]
	cell_arr=domain.geometry.dofmap.array[:]
	#the node mappings
	old_2_new_node = np.array(domain.geometry.input_global_indices)
	new_2_old_node = np.argsort(old_2_new_node)

	#loop through dofmap of new elements
	#and match new element number with old element number
	#we want the old_2_new cell mapping
	old_2_new_cell = np.zeros(ncell,dtype=np.int32)

	#get list of new node nums from the new dofmap
	new_node_num_old_nm = new_2_old_node[nm.flatten()].reshape(nm.shape)

	#to compare, sort each of the element connectivities
	new_node_num_new_nm = domain.geometry.dofmap.array[:].reshape(nm.shape)

	sort_new_nm = np.sort(new_node_num_new_nm,axis=1)

	#print("new node numbering with old connectivity")
	sort_old_nm = np.sort(new_node_num_old_nm,axis=1)
	
	#use original nm to find old_2_new_cell mapping
	#print(np.sort(nm,axis=1))

	#find matching entries and store in old_2_new_cell
	#brute force for now but there is better
	#algorithms out there
	for i,old_cell in enumerate(sort_old_nm):
		#find which new cell number matches
		for j in range(ncell):
			if np.array_equal(old_cell,sort_new_nm[j]):
				#print("old cell",i," matches with new cell",j)
				old_2_new_cell[i] = j
	return old_2_new_cell


		

#this gives us a map from old cell numbers to new
old_2_new = find_element_mapping(domain,coords,nm)




'''
#This seems to reorder everything
#print(domain.geometry.x)
# show nodal order--adjusted back to input

#print(idcs)
#print(domain.geometry.x[idcs,:])
#can we overwrite?


domain.geometry.x[:,:]=domain.geometry.x[idcs,:]
#domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim)
#something. like this?
#dests = np.full(topo.num_nodes, comm.rank, dtype=np.int32)
#offsets = np.arange(topo.num_nodes+1, dtype=np.int32)
#dolfinx.graph.create_adjacencylist(dests, offsets)


#adj = adjacencylist(nm)


domain.geometry.dofmap.array[:] = nm.flatten()

#print(domain.geometry.dofmap.array[:])
#print(nm)
#domain.topology.index_map(2)


#domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim)
#domain.topology.create_connectivity(domain.topology.dim-1, domain.topology.dim)
#domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim-1)
#domain.topology.create_connectivity(domain.topology.dim-1, domain.topology.dim-1)

#domain.topology.set_connectivity(adj,2,0)

meshconnectivity=domain.geometry.dofmap
domain.geometry.input_global_indices[:]=np.arange(len(domain.geometry.input_global_indices))

for i in range(domain.topology.dim):
        domain.topology.create_entities(i)
        for j in range(domain.topology.dim):
        	domain.topology.create_connectivity(i, j)

#overwrite maybe?

#Why can't we do this?
#domain.topology.set_connectivity(domain.geometry.dofmap,2,2)

#print(nm)
#print(meshconnectivity)
#plot mesh to see if it works
#xdmf = io.XDMFFile(domain.comm, filename+"/"+filename+"_mesh.xdmf", "w")
#xdmf.write_mesh(domain)
#xdmf.close()




'''
print(domain.geometry.x)
print(domain.geometry.dofmap)

####################################################################################
###################################################################################
##########Define Length of simulation and time step size##########################
#ts is start time in seconufl.ds
ts=0.0
#tf is final time in seconufl.ds
tf=60*60
#time step size in seconufl.ds
dt=60
#####################################################################################
####We need to identify function spaces before we can assign initial conditions######

#We will use "DG" elements
p_type = "DG"
#polynomial order of finite element, h is first , (u,v) is second
p_degree = [1,1]


# We use mixed elements, these are ufl objects for symbolic math
el_h   = ufl.FiniteElement(p_type, domain.ufl_cell(), degree=p_degree[0])
el_vel = ufl.VectorElement(p_type, domain.ufl_cell(), degree=p_degree[1], dim = 2)


#V will hold the function space info of the mixed elements
V = fe.FunctionSpace(domain, ufl.MixedElement([el_h, el_vel]))

V0 = fe.FunctionSpace(domain, ("Discontinuous Lagrange", 0))
#Let's see if we can find stuff
#need to be able to connect cell -> dof
#what node #s for each cell

#Centroids of each cell
DG1_DOFS=V.sub(0).collapse()[0].tabulate_dof_coordinates()
#print(DG1_DOFS)
DG1_V_DOFS=V.sub(0).collapse()[0].tabulate_dof_coordinates()


#solution variables
#this will store solution as we march through time
u = fe.Function(V)

#split into h, and velocity
h, vel = u.split()

#also solutions for previous time steps
u_n = fe.Function(V)
u_n_old = fe.Function(V)
#function that stores any dirichlet boundary conditions
u_ex = fe.Function(V)

#Create test functions
p1, p2 = ufl.TestFunctions(V)
# object that concatenates all test functions into single variable, like u
p = ufl.as_vector((p1,p2[0],p2[1]))

################################################################################
################################################################################

####Assigning bathymetry and initial conditions#########
##Note that fenicsx can handle bathymetry from different function space
#But we will keep same for now

#Bathymetry assignment

#for this problem, assume uniform depth of 10 m
h_b = fe.Function(V.sub(0).collapse()[0])
#Event though constant still neeufl.ds to be function of x by convention
h_b.interpolate(lambda x: depth + 0*x[0])


#Initial condition assignment for up/down flow
#in this case, initial condition is h=h_b, vel=(vel_boundary_mag,0)
#introduce a shock to system to mess with conditioning
perturb_vel=1.0

'''
u_n.sub(0).interpolate(
	fe.Expression(
		h_b, 
		V.sub(0).element.interpolation_points()))
  
u_n.sub(1).interpolate(
	fe.Expression(
		ufl.as_vector([fe.Constant(domain, ScalarType(0.0)),
			fe.Constant(domain, ScalarType(vel_boundary_mag))]),
		V.sub(1).element.interpolation_points()))

#also need to input bathymetry to u_ex to store h_b
h_ex = u_ex.sub(0)
h_ex.interpolate(
	fe.Expression(
		h_b,
		V.sub(0).element.interpolation_points())
	)
vel_ex = u_ex.sub(1)
vel_ex.interpolate(
	fe.Expression(
		ufl.as_vector([fe.Constant(domain, ScalarType(perturb_vel)),
			fe.Constant(domain, ScalarType(vel_boundary_mag))]),
		V.sub(1).element.interpolation_points()))

'''
#'''
#Initial condition assignment for l/r flow
#in this case, initial condition is h=h_b, vel=(vel_boundary_mag,0)
#introduce a shock to system to mess with conditioning
#perturb flow vertically


u_n.sub(0).interpolate(
	fe.Expression(
		h_b, 
		V.sub(0).element.interpolation_points()))
  
u_n.sub(1).interpolate(
	fe.Expression(
		ufl.as_vector([fe.Constant(domain, ScalarType(vel_boundary_mag)),
			fe.Constant(domain, ScalarType(0.0))]),
		V.sub(1).element.interpolation_points()))

#also need to input bathymetry to u_ex to store h_b
h_ex = u_ex.sub(0)
h_ex.interpolate(
	fe.Expression(
		h_b,
		V.sub(0).element.interpolation_points())
	)
vel_ex = u_ex.sub(1)
vel_ex.interpolate(
	fe.Expression(
		ufl.as_vector([fe.Constant(domain, ScalarType(vel_boundary_mag)),
			fe.Constant(domain, ScalarType(perturb_vel))]),
		V.sub(1).element.interpolation_points()))
#'''



################################################################################
################################################################################

#####Time dependent boundary conditions######
#####Define boundary condition locations#####

# 1 is the left side, and will be an inflow boundary condition (Dirichlet condition for surface elevation and velocity)
# 3,4 top and bottom sides are no flux condition (U \cdot n = 0)
# 2 is free outflow condition
# We can add more numbers for different bc types later

#this ligns flow down to up, matching with cell numbering
'''
boundaries = [(1, lambda x: np.isclose(x[1], y0)),
              (2, lambda x: np.isclose(x[1], y1)),
              (3, lambda x: np.isclose(x[0], x0)),
              (4, lambda x: np.isclose(x[0], x1))]
'''
#'''
#this aligns flow left to right, now line numbers don't agree with cell numbering
boundaries = [(1, lambda x: np.isclose(x[0], x0)),
              (2, lambda x: np.isclose(x[0], x1)),
              (3, lambda x: np.isclose(x[1], y0)),
              (4, lambda x: np.isclose(x[1], y1))]

#'''
##########Defining functions which actually apply the boundary conditions######
facet_markers, facet_tag = MarkBoundary(domain, boundaries)

#generate a measure with the marked boundaries
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)


##########Dirchlet Boundary conditions###################
# Define the boundary conditions and pass them to the solver
dirichlet_conditions = []


#identify equation numbers associated with boundary conditions
#this is only necessary for dirichlet conditions
#can add in more later
#the dirichlet_conditions are a list containing dolfinx functions that assign bc
for marker, func in boundaries:
	if marker == 1:
		h_dirichlet_dofs,bc = BoundaryCondition("Open", marker, func, u_ex.sub(0), V.sub(0))
		dirichlet_conditions.append(bc)
		vel_dirichlet_dofs,bc = BoundaryCondition("Open", marker, func, u_ex.sub(1), V.sub(1))
		dirichlet_conditions.append(bc)

'''
print("Length of solution vector",len(u_n.x.array[:]))
print("the coordinates of bc for h", h_dirichlet_dofs)
print("the coordinates of bc for vel", vel_dirichlet_dofs)
exit(0)
'''

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
vel_boundary = vel_ex.x.array[vel_dirichlet_dofs]  

################################################################################
################################################################################

#######Establish weak form to solve within time loop########

#aliasing to make reading easier
h, ux, uy = u[0], u[1], u[2]
#also shorthand reference for flux variable:
Q      =   ufl.as_vector((u[0], u[1]*u[0], u[2]*u[0] ))
Qn     =   ufl.as_vector((u_n[0], u_n[1]*u_n[0], u_n[2]*u_n[0]))
Qn_old =   ufl.as_vector((u_n_old[0], u_n_old[1]*u_n_old[0], u_n_old[2]*u_n_old[0] )) 


#g is gravitational constant
g=9.81


#Flux tensor from SWE
Fu = ufl.as_tensor([[h*ux,h*uy], 
				[h*ux*ux+ 0.5*g*h*h-0.5*g*h_b*h_b, h*ux*uy],
				[h*ux*uy,h*uy*uy+0.5*g*h*h-0.5*g*h_b*h_b]
				])

#Flux tensor for SWE if normal flow is 0
Fu_wall = ufl.as_tensor([[0,0], 
					[0.5*g*h*h-0.5*g*h_b*h_b, 0],
					[0,0.5*g*h*h-0.5*g*h_b*h_b]
					])


#RHS source vector for SWE is gravity + bottom friction
#can add in things like wind and pressure later
g_vec = ufl.as_vector((0,
 					-g*(h-h_b)*h_b.dx(0),
 					-g*(h-h_b)*h_b.dx(1)))
#there are many friction laws, here is an example of a quadratic law
#which matches an operational model ADCIRC
'''
eps=1e-8
mag_v = ufl.conditional(pow(ux*ux + uy*uy, 0.5) < eps, eps, pow(ux*ux + uy*uy, 0.5))
FFACTOR = 0.0025
HBREAK = 1.0
FTHETA = 10.0
FGAMMA = 1.0/3.0
Cd = ufl.conditional(h>eps, (FFACTOR*(1+HBREAK/h)**FTHETA)**(FGAMMA/FTHETA), eps  )
fric_vec = as_vector((0,
					Cd*ux*mag_v,
					Cd*uy*mag_v))

'''
#Linear friction law
cf=0.0001
fric_vec=ufl.as_vector((0,
                    ux*cf,
                    uy*cf))

#try no friction
S = g_vec+fric_vec



#normal vector
n = ufl.FacetNormal(domain)




#begin constructing the weak form, this is standard weak form from IBP
#start adding to residual, beggining with body term
F = -ufl.inner(Fu,ufl.grad(p))*ufl.dx
#add RHS forcing
F += ufl.inner(S,p)*ufl.dx


#now adding in global boundary terms
for marker, func in boundaries:
	if (marker == 1) or (marker == 2):
		#This is the open boundary in this case
		F += ufl.dot(ufl.dot(Fu, n), p) * ds(marker)
	else:
		print("Adding wall condition \n\n")
		#this is the wall condition, no flux on this part
		F += ufl.dot(ufl.dot(Fu_wall, n), p)*ds(marker)

#now adding interior boundary terms using Lax-Friedrichs upwinding for DG
eps=1e-8
#attempt at full expression from https://docu.ngsolve.org/v6.2.1810/i-tutorials/unit-3.4-simplehyp/shallow2D.html
vela =  ufl.as_vector((u[1]('+'),u[2]('+')))
velb =  ufl.as_vector((u[1]('-'),u[2]('-')))
vnorma = ufl.conditional(ufl.sqrt(ufl.dot(vela,vela)) > eps,ufl.sqrt(ufl.dot(vela,vela)),eps)
vnormb = ufl.conditional(ufl.sqrt(ufl.dot(velb,velb)) > eps,ufl.sqrt(ufl.dot(velb,velb)),eps)
C = ufl.conditional( (vnorma + ufl.sqrt(g*u[0]('+'))) > (vnormb + ufl.sqrt(g*u[0]('-'))), (vnorma + ufl.sqrt(g*u[0]('+'))) ,  (vnormb + ufl.sqrt(g*u[0]('-')))) 
flux = ufl.dot(ufl.avg(Fu), n('+')) + 0.5*C*ufl.jump(Q)

F += ufl.inner(flux, ufl.jump(p))*ufl.dS


#now add terms related to time step
#specifies time stepping scheme, save it as fe.constant so it is modifiable
theta=1
theta1 = fe.Constant(domain, ScalarType(theta))


# this is a generalized version of the BDF2 scheme
#theta1=0 is 1st order implicit Euler, theta1=1 is 2nd order BDF2
dQdt = theta1*fe.Constant(domain,ScalarType(1.0/dt))*(1.5*Q - 2*Qn + 0.5*Qn_old) + (1-theta1)*fe.Constant(domain,ScalarType(1.0/dt))*(Q - Qn)

#add to weak form
#try different form for preconditioner
F+=ufl.inner(dQdt,p)*ufl.dx
#Fpre=F-ufl.inner((theta1*fe.Constant(domain,ScalarType(1.0/dt))*( - 2*Qn + 0.5*Qn_old) + (1-theta1)*fe.Constant(domain,ScalarType(1.0/dt))*(- Qn)),p)*ufl.dx



#Weak form and initial conditions are now arranged, 
# Almost ready for Main time loop but need to initialize a few thing

###################################################################################
###################################################################################
#Some final steps before the main time loop
#Uses FEniCSx to advance time by using Newton-Raphson for each implicit time step

#First let's create plots so we can view initial condition before advancing in time
xdmf = io.XDMFFile(domain.comm, filename+"/"+filename+".xdmf", "w")
xdmf.write_mesh(domain)

#initiate some auxillary functions for plotting
#it is more common to plot surface elevation (eta) rather than depth (h)        
V_scalar = V.sub(0).collapse()[0]
V_vel = V.sub(1).collapse()[0]

eta_plot = fe.Function(V_scalar)
eta_plot.name = "WSE(m)"


vel_plot = fe.Function(V_vel)
vel_plot.name = "depth averaged velocity (m/s)"

def plot_global_output(u,h_b,V_scalar,V_vel,xdmf,t):
	#interpolate and plot water surface elevation and velocity
	eta_expr = fe.Expression(u.sub(0).collapse() - h_b, V_scalar.element.interpolation_points())
	eta_plot.interpolate(eta_expr)
	v_expr = fe.Expression(u.sub(1).collapse(), V_vel.element.interpolation_points())
	vel_plot.interpolate(v_expr)

	xdmf.write_function(eta_plot,t)
	xdmf.write_function(vel_plot,t)
	return 0

#print("Length of solution = ",u_n.x.array[:])
plot_global_output(u_n,h_b,V_scalar,V_vel,xdmf,ts)

#######Initialize a solver object###########
#create the line smoothing preconditioner, assuming static for now. Later this will need to go inside time loop
#for now we know this map but later we will create a routine that creates the order of elements in each line
def find_lines(u,domain):
	#the solution and domain should be all we need?
	#this should return a set of lists with each list containing element numbers in each streamline

	#for now it's hard coded, we know that we have 2 lines up/down flow
	#return [[0,1],[2,3]]
	#if we have l/r flow
	#return [[0,2],[1,3]]

	#assuming l/r then the lines are 2x nx and then use our map to translate
	return np.arange(nx*ny*2).reshape(ny,nx*2)
	#this is u/d orthogonal for comparison
	#return np.arange(nx*ny*2).reshape(ny,nx*2).T

#call the routine to get set of lists
lines = find_lines(u,domain)
print(lines)
print("new lines order")
lines = lines.flatten()[old_2_new].reshape(ny,nx*2)
print(lines)

#utilize the custom Newton solver class instead of the fe.petsc Nonlinear class
#mesh and lines inputs are specifically for line smoothing preconditioner

#standard solver
#DG FEM -> linearization
#Newton_Solver = CustomNewtonProblem(F,u,dirichlet_conditions, domain.comm, solver_parameters=params,mesh=domain,lines=lines,Fpre=F)


#alternative solver -> linearization -> DG FEM
Newton_Solver = CustomNewtonProblem2(Fu,Fu_wall,S,dQdt,u,p,n,ds,boundaries,dirichlet_conditions, domain.comm, solver_parameters=params,mesh=domain,lines=lines,Fpre=F)



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
start = time.time()
for a in range(min(2,nt)):
	#by default we print out on screen each time step
	print('Time Step Number',a,'Out of',nt)
	print(a/nt*100,'% Complete')
	#save new solution as previous solution
	u_n_old.x.array[:] = u_n.x.array[:]
	u_n.x.array[:] = u.x.array[:]
	#update time
	t += dt
	#update any dirichlet boundary conditions
	#for now just the one but may expand in future
	u_ex.sub(0).x.array[h_dirichlet_dofs] = update_boundary(t,hb_boundary)
	u_ex.sub(1).x.array[vel_dirichlet_dofs]=vel_boundary
	u.x.array[h_dirichlet_dofs] = u_ex.x.array[h_dirichlet_dofs]
	u.x.array[vel_dirichlet_dofs] = u_ex.x.array[vel_dirichlet_dofs]
	#solve associated NewtonProblem
	Newton_Solver.solve(u)
	#add data to station variable
	station_data[a+1,:,:] = record_stations(u,local_points,local_cells)
	#Plot global solution
	if (a+1)%plot_every==0 and plot_every <= nt:
		plot_global_output(u,h_b,V_scalar,V_vel,xdmf,t)

#Take remainder of time steps with 2nd order BDF2 scheme
theta1.value=0
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
	u_ex.sub(1).x.array[vel_dirichlet_dofs]=vel_boundary
	u.x.array[h_dirichlet_dofs] = u_ex.x.array[h_dirichlet_dofs]
	u.x.array[vel_dirichlet_dofs] = u_ex.x.array[vel_dirichlet_dofs]
	#solve associated NewtonProblem
	Newton_Solver.solve(u)
	#add data to station variable
	station_data[a+1,:,:] = record_stations(u,local_points,local_cells)
	#Plot global solution
	if (a+1)%plot_every==0:
		print("Plotting solution for t = ", str(t/3600.0),"hr")
		plot_global_output(u,h_b,V_scalar,V_vel,xdmf,t)

print("time loop takes ", time.time()-start)
#################################################################################
#################################################################################

#Time loop is complete, any postprocessing may go here
if plot_every <= nt:
	xdmf.close()

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

