import numpy as np
from dolfinx import fem as fe, mesh,io
from mpi4py import MPI
from ufl import (
    VectorElement, TestFunction, TrialFunction, FacetNormal, as_matrix,
    as_vector, as_tensor, dot, inner, grad, dx, ds, dS,
    jump, avg,sqrt,conditional,gt,div,nabla_div,tr,diag,sign,elem_mult,
    MixedElement, FiniteElement, TestFunctions, Measure
)
from petsc4py.PETSc import ScalarType
from boundaryconditions import BoundaryCondition,MarkBoundary

###########################################
#######Define the physical domain########
#For this basic script, just a rectangle

#First define physical dimensions
#Coordinate of bottom left corner
x0 = 0.0
y0 = 0.0
#Coordinate of top right corner
x1= 10000.0
y1= 2000.0


#Now define mesh properties
#number of cells in x and y direction
nx=20
ny=5

#creates dolfinx mesh object partioned via MPI
domain = mesh.create_rectangle(MPI.COMM_WORLD, [[x0, y0],[x1, y1]], [nx, ny])


####################################################################################

####We need to identify function spaces before we can assign initial conditions######

#We will use "DG" elements
p_type = "DG"
#polynomial order of finite element, h is first , (u,v) is second
p_degree = [1,1]

# We use mixed elements, these are ufl objects for symbolic math
el_h   = FiniteElement(p_type, domain.ufl_cell(), degree=p_degree[0])
el_vel = VectorElement(p_type, domain.ufl_cell(), degree=p_degree[1], dim = 2)


#V will hold the function space info of the mixed elements
V = fe.FunctionSpace(domain, MixedElement([el_h, el_vel]))

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
p1, p2 = TestFunctions(V)
# object that concatenates all test functions into single variable, like u
p = as_vector((p1,p2[0],p2[1]))


################################################################################

####Assigning bathymetry and initial conditions#########
##Note that fenicsx can handle bathymetry from different function space
#But we will keep same for now

#Bathymetry assignment

#for this problem, assume uniform depth of 10 m
depth=10.0
h_b = fe.Function(V.sub(0).collapse()[0])
#Event though constant still needs to be function of x by convention
h_b.interpolate(lambda x: depth + 0*x[0])


#Initial condition assignment
#in this case, initial condition is h=h_b, vel=(0,0)
u_n.sub(0).interpolate(
	fe.Expression(
		h_b, 
		V.sub(0).element.interpolation_points()))
  
u_n.sub(1).interpolate(
	fe.Expression(
		as_vector([fe.Constant(domain, ScalarType(0.0)),
			fe.Constant(domain, ScalarType(0.0))]),
		V.sub(1).element.interpolation_points()))

#also need to input bathymetry to u_ex function which forces dirichlet b.c.
h_ex = u_ex.sub(0)
h_ex.interpolate(
	fe.Expression(
		h_b,
		V.sub(0).element.interpolation_points())
	)

################################################################################

#####Time dependent boundary conditions#############


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
	mag = 0.15
	return mag*np.cos(t*alpha)

#define h_b at boundary of dof so we don't need to repeat in time loop
hb_boundary = h_ex.x.array[h_dirichlet_dofs]
#A function which will update the dirichlet bc inside the time loop
def update_boundary(t,hb):
	#take in time and return float of tide level
	tide_level = evaluate_tidal_boundary(t)
	#return np array with tide level at dirichlet boundary
	return hb_boundary + tide_level

#should be able to assign via u_ex.sub(0).x.array[self.dof_open] = bc


###############################################################################

#######Establish weak form to solve within time loop########

#aliasing to make reading easier
h, ux, uy = u[0], u[1], u[2]

#g is gravitational constant
g=9.81


#Flux tensor from SWE
Fu = as_tensor([[h*ux,h*uy], 
				[h*ux*ux+ 0.5*g*h*h-0.5*g*h_b*h_b, h*ux*uy],
				[h*ux*uy,h*uy*uy+0.5*g*h*h-0.5*g*h_b*h_b]
				])



#RHS source vector for SWE is gravity + bottom friction
#can add in things like wind and pressure later
g_vec = as_vector((0,
 					-g*(h-h_b)*h_b.dx(0),
 					-g*(h-h_b)*h_b.dx(1)))
#there are many friction laws, here is an example of a quadratic law
#which matches an operational model ADCIRC
eps=1e-8
mag_v = conditional(pow(ux*ux + uy*uy, 0.5) < eps, eps, pow(ux*ux + uy*uy, 0.5))
FFACTOR = 0.0025
HBREAK = 1.0
FTHETA = 10.0
FGAMMA = 1.0/3.0
Cd = conditional(h>eps, (FFACTOR*(1+HBREAK/h)**FTHETA)**(FGAMMA/FTHETA), eps  )
fric_vec = as_vector((0,
					Cd*ux*mag_v,
					Cd*uy*mag_v))

S = g_vec+fric_vec



#normal vector
n = FacetNormal(domain)
#specifies time stepping scheme, save it as fe.constant so it is modifiable
theta=1
theta1 = fe.Constant(domain, PETSc.ScalarType(theta))



#begin constructing the weak form, this is standard weak form from IBP
#start adding to residual, beggining with body term
F = -inner(Fu,grad(p))*dx

#now adding boundary terms
###STOPPING HERE####
for condition in boundary_conditions:
	if condition.type == "Open":
                self.F += dot(dot(self.Fu, n), self.p) * ds_exterior(condition.marker)

            if condition.type == "Wall":
                self.F += dot(dot(self.Fu_wall, n), self.p)*ds_exterior(condition.marker)
#now adding DG


