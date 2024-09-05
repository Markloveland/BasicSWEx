from pathlib import Path
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
from dolfinx import fem as fe, mesh,io
from dolfinx import __version__
from mpi4py import MPI
from ufl import (
    TestFunction, TrialFunction, FacetNormal, as_matrix,
    as_vector, as_tensor, dot, inner, grad, dx, ds, dS,
    jump, avg,sqrt,conditional,gt,div,nabla_div,tr,diag,sign,elem_mult,
    TestFunctions, Measure
)
from basix.ufl import element, mixed_element
from dolfinx.fem import functionspace
from dolfinx.fem.petsc import LinearProblem
import dolfinx.fem.petsc as petsc
from petsc4py.PETSc import ScalarType
from boundaryconditions import BoundaryCondition,MarkBoundary
from newton import CustomNewtonProblem
from auxillaries import (init_stations, record_stations, gather_stations, get_F, get_LF_flux_form,
compute_cell_boundary_facets, compute_interior_facet_integration_entities, get_interior_facets,
compute_norm_special)
from dolfinx.cpp.mesh import cell_num_entities
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from petsc4py import PETSc
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
#Coordinate of top right corner
y1= 2000.0
x1= 2000.0


#Now define mesh properties
#number of cells in x and y direction
nx=10
ny=10#5

#Propagation velocity entering the left side
#A proper bc should be flux maybe?
depth = 2
vel_boundary_mag = 9.0
flux_boundary_mag = depth*vel_boundary_mag

#creates dolfinx mesh object partioned via MPI
domain = mesh.create_rectangle(MPI.COMM_WORLD, [[x0, y0],[x1, y1]], [nx, ny])

####################################################################################
####################################################################################


###################################################################################
##########Define Length of simulation and time step size##########################
#ts is start time in seconds
ts=0.0
#tf is final time in seconds
tf=3600#7*24*60*60
#time step size in seconds
dt=60.0
#####################################################################################
####We need to identify function spaces before we can assign initial conditions######

#We will use "DG" elements
p_type = "DG"
#polynomial order of finite element, h is first , (u,v) is second
p_degree = [1,1]

# Because we are evaluating fluxes, this must have 080
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
h_b = fe.Function(V_scalar)
#also velocity assignment
perturb_vel=0.0

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
		as_vector([fe.Constant(domain, ScalarType(vel_boundary_mag)),
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
		as_vector([fe.Constant(domain, ScalarType(vel_boundary_mag)),
			fe.Constant(domain, ScalarType(perturb_vel))]),
		V.sub(1).element.interpolation_points()))

################################################################################
################################################################################

################################################################################
################################################################################
#Initialize Line Preconditioner

#use mixed domain to evaluate inter-elemental fluxes
#this will then be used to create lines
#try evaulating interelemental fluxes with mixed domain thing
#requires version > 090
# Create a sub-mesh of all facets in the mesh to allow the facet function
# spaces to be created
tdim = domain.topology.dim
fdim = tdim - 1
num_cell_facets = cell_num_entities(domain.topology.cell_type, fdim)
#create facet connectivities
domain.topology.create_entities(fdim)
#save facet imap
facet_imap = domain.topology.index_map(fdim)
num_facets = facet_imap.size_local + facet_imap.num_ghosts
facets = np.arange(num_facets, dtype=np.int32)
# NOTE Despite all facets being present in the submesh, the entity map isn't
# necessarily the identity in parallel
facet_mesh, facet_mesh_to_msh = mesh.create_submesh(domain, fdim, facets)[0:2]
msh_to_facet_mesh = np.full(num_facets, -1)
msh_to_facet_mesh[facet_mesh_to_msh] = np.arange(len(facet_mesh_to_msh))
#use this to map facets on facet mesh to mother mesh
entity_maps = {facet_mesh: msh_to_facet_mesh}
# Create functions spaces
k_trace = 0  # Polynomial degree
#DG that lives on mesh skeleton
Vbar = fe.functionspace(facet_mesh, ("Discontinuous Lagrange", k_trace))
ubar, vbar = TrialFunction(Vbar), TestFunction(Vbar)
# Create integration entities and define integration measures. We want
# to integrate around each element boundary, so we call the following
# convenience function:
cell_boundary_facets = compute_cell_boundary_facets(domain)
cell_imap = domain.topology.index_map(tdim)
num_cells = cell_imap.size_local + cell_imap.num_ghosts
#retruns facets in three different arrays
facet_locs,interior_facets_plus, interior_facets_minus, exterior_facets  = get_interior_facets(domain,np.arange(num_cells))
dS_pre = Measure(
    "ds",
    domain=domain,
    subdomain_data=[
        (0, interior_facets_plus),
        (1, interior_facets_minus),
        (2, exterior_facets)
    ],
)



# Create a cell integral measure over the facet mesh for L2 projection
dx_f = Measure("dx", domain=facet_mesh)


#Perform an L2 projection for each entry
#now see if we can construct an integral over each facet to create projection
A = ubar*vbar*dx_f
Aform = fe.form(A,entity_maps=entity_maps)
Amat = petsc.assemble_matrix(Aform)
Amat.assemble()


#always restrict to positive, instead switch with 0 and +
#only way to do conditional is to integrate and then take max vector
#there is error in C but good enough most likely
n = FacetNormal(domain)
#this will be actual solution, but we already assigned initial condition so use this for now
Fu_pre = get_F(u_n,h_b)
#do one equation at a time and then take some sort of norm
ndof_trace = Vbar.dofmap.index_map.size_local + Vbar.dofmap.index_map.num_ghosts

#ufl forms for each equation
L_continuity = get_LF_flux_form(Fu_pre,u_n,n,vbar,dS_pre(0),dS_pre(1), dS_pre(2),0)
L_momentum_x = get_LF_flux_form(Fu_pre,u_n,n,vbar,dS_pre(0),dS_pre(1), dS_pre(2),1)
L_momentum_y = get_LF_flux_form(Fu_pre,u_n,n,vbar,dS_pre(0),dS_pre(1), dS_pre(2),2)
#turn into aseemblable forms
Lform_continuity = fe.form(L_continuity,entity_maps=entity_maps)
Lform_momemntum_x = fe.form(L_momentum_x,entity_maps=entity_maps)
Lform_momemntum_y = fe.form(L_momentum_y,entity_maps=entity_maps)

#assemble and store in vectors
Lvec_continuity = petsc.assemble_vector(Lform_continuity)
Lvec_momentum_x = petsc.assemble_vector(Lform_momemntum_x)
Lvec_momentum_y = petsc.assemble_vector(Lform_momemntum_y)

Lvec_continuity.assemble()
Lvec_momentum_x.assemble()
Lvec_momentum_y.assemble()

#create one solver for L2 Projection
ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(Amat)
ksp.setType("preonly")
ksp.getPC().setType("lu")
# Compute solution
x_con = Amat.createVecRight()
x_momx = Amat.createVecRight()
x_momy = Amat.createVecRight()
#solve
ksp.solve(Lvec_continuity, x_con)
ksp.solve(Lvec_momentum_x, x_momx)
ksp.solve(Lvec_momentum_y, x_momy)


nrow = Lvec_continuity.getSize()
print("L2 interpolation to facets continuity",x_con.array)
print("L2 interpolation to facets mom-x",x_momx.array)
print("L2 interpolation to facets mom-y",x_momy.array)

#compute the norm between the arrays
facet_flux = compute_norm_special(x_con.array,x_momx.array,x_momy.array,np.inf)

#plots stuff
#with io.VTXWriter(domain.comm, "u.bp", h_b, "bp4") as f:
#    f.write(0.0)

#this routine finds cell pairs that are linked above a certain threshold
def get_cell_pairs(domain,facet_flux,fdim,tdim,tol=1e-2):
	#first get indices of nonzero fluxes
	f_mask = np.where(facet_flux>tol)[0]
	f_to_c = domain.topology.connectivity(fdim, tdim)

	cell_dat = []
	#iterate through facets and keep track of cells
	for i in f_mask:
		cell_dat.append(f_to_c.links(i))
	return np.array(cell_dat)



#this one doesnt really work
def get_lines(cell_pairs,ncells):
	#simple algorithm
	#takes in list of cell pairs and returns sets of lines
	print("forming lines")
	temp_cell_pairs = np.empty_like(cell_pairs)
	cell_pairs.sort()
	temp_cell_pairs[:] = cell_pairs
	#remaining_ind = np.arange(ncells)
	ctr = 0
	lines = []
	ncell_count = 0
	nline = 0
	while (len(temp_cell_pairs)!=0):
		ctr=0
		#initialize current line, remove from array
		current_line = [temp_cell_pairs[ctr,0],temp_cell_pairs[ctr,1]]
		current_pair = temp_cell_pairs[ctr]
		temp_cell_pairs = np.delete(temp_cell_pairs,ctr,axis=0)
		if(len(temp_cell_pairs)==0):
			break
		#check intersections of remaining facets, add it if nonempty
		linecomplete=True

		
		while(linecomplete):
			check_pair = temp_cell_pairs[ctr]
			#for a in remaining_pairs:
			if np.intersect1d(current_pair,check_pair).size>0:
				#pair found, update lines
				current_line.append(check_pair[0])
				current_line.append(check_pair[1])
				#remove pair from list
				temp_cell_pairs = np.delete(temp_cell_pairs,ctr,axis=0)
				if(len(temp_cell_pairs)==0):
					tmp = np.unique(current_line)
					ncell_count+=tmp.size
					lines.append(tmp)
					nline+=1
					linecomplete=False
				#move update current pair and start over
				current_pair = check_pair
				ctr = 0

			else:
				ctr+=1
				if ctr == len(temp_cell_pairs):
					#append this to lines, eliminate repeats
					tmp = np.unique(current_line)
					ncell_count+=tmp.size
					lines.append(tmp)
					current_line = []
					linecomplete=False
					nline+=1

	#if all cells aren't included, append to end cell by cell
	if ncell_count != num_cells:
		print("WARNING, not all cells strongly connected")
		#find missing cell numbers and append ...
		#will add this later
	else:
		print("All cells in a line")
	return lines

def MakePath(elem_no, c_to_f, f_to_c, f_flux, line,boundary_facets):
	#for element number, pick face with highest connectivity
	facet_nos = c_to_f.links(elem_no)
	facet_fluxes = f_flux[facet_nos]
	#pick facets that with highest 2 connectivities
	temp = facet_nos[np.argsort(facet_fluxes)]
	max_facs = temp[-2:]
	#check if facet is boundary facet
	#i dont think we want to terminate here
	#if np.isin(max_facs[-1],boundary_facets):
	#	print("HIT BOUNDARY FACET")
	#	print(facet_fluxes)
	#try highest cell first
	cell_nos = f_to_c.links(max_facs[-1])
	#see what cell nos are not already in line
	new_cell = np.setdiff1d(cell_nos, line)
	#if new cell is empty then notihng gets added to path
	if len(new_cell) !=0:
		k = new_cell[0]
		line.append(k)
	else:
		#search for second highest one
		cell_nos = f_to_c.links(max_facs[-2])
		#see what cell nos are not already in line
		new_cell = np.setdiff1d(cell_nos, line)
		if len(new_cell) !=0:
			k = new_cell[0]
			line.append(k)
		else:
			#terminate path
			k=-1
	return line,k
#from dissertation
def LineCreation(seed_element,c_to_f, f_to_c, flux,boundary_facets):
	line = [seed_element]
	k =seed_element
	while k!=-1:
		print(k)
		line,k_new = MakePath(k, c_to_f, f_to_c, flux,  line,boundary_facets)
		k=k_new
	return line

def GenerateLines(cells,c_to_f,f_to_c,flux,boundary_facets):
	lines = []
	while cells.size>0:
		seed_element = cells[0]
		line = LineCreation(seed_element,c_to_f, f_to_c, flux,boundary_facets)
		#remove all cells from this list
		mask = np.isin(cells, line, invert=True)
		cells = cells[mask]
		#append line to lines
		lines.append(line)
	return lines



boundary_facets = mesh.exterior_facet_indices(domain.topology)
#try making a line with one seed element and see what happens
f_to_c = domain.topology.connectivity(fdim, tdim)
c_to_f = domain.topology.connectivity(tdim, fdim)
cells = np.arange(num_cells)
lines = GenerateLines(cells,c_to_f,f_to_c,facet_flux,boundary_facets)
DG0 = fe.functionspace(domain, ("DG", 0))
cell_centroids = DG0.tabulate_dof_coordinates()
#print("FIRST CENTROID", cell_centroids[0])
lines = np.array(lines)



#######################################
# Plot Lines of initial condition
#lets see if we can visualize lines somehow
nds = domain.geometry.x
ei = domain.geometry.dofmap
tris = tri.Triangulation(nds[:,0], nds[:,1], triangles=ei)
#assign mat
mat = np.zeros(ei.shape[0])
nline = 0
for a,line in enumerate(lines):
	mat[line] = a
	nline+=1
fig = plt.figure(figsize=(18, 12),facecolor=(1, 1, 1))
plt.tripcolor(tris, facecolors=mat, edgecolors='k',cmap=plt.cm.get_cmap('rainbow', nline),linewidth=1)
cbar = plt.colorbar(ticks=np.arange(np.min(mat), np.max(mat) + 1))
cbar.ax.tick_params(labelsize=20)
plt.xlabel('x',fontsize=20)
plt.ylabel('y',fontsize=20)
plt.title("Material Plot",fontsize=20)

#also plot arrows of each line
for a,line in enumerate(lines):
	plt.plot(cell_centroids[line,0],cell_centroids[line,1])

plt.savefig("Lines.png")

################################################################################
################################################################################

################################################################################
################################################################################
#####Time dependent boundary conditions######
#####Define boundary condition locations#####

# 1 is the left side, and will be a tidal boundary condition (Dirichlet condition for surface elevation)
# 2 are all other sides and are no flux condition (U \cdot n = 0)
# We can add more numbers for different bc types later
boundaries = [(1, lambda x: np.isclose(x[0], x0)),
              (2, lambda x: np.isclose(x[0], x1)),
              (3, lambda x: np.isclose(x[1], y0)),
              (4, lambda x: np.isclose(x[1], y1))]

##########Defining functions which actually apply the boundary conditions######
facet_markers, facet_tag = MarkBoundary(domain, boundaries)
#generate a measure with the marked boundaries
ds = Measure("ds", domain=domain, subdomain_data=facet_tag)


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



###To prepare for time loop, assign the boundary forcing function, u_ex
#this will compute the tidal elevation at the boundary
def evaluate_tidal_boundary(t):
	#hard coded parameters for mag and frequency
	alpha = 0.00014051891708
	mag = 0.0
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
Q      =   as_vector((u[0], u[1]*u[0], u[2]*u[0] ))
Qn     =   as_vector((u_n[0], u_n[1]*u_n[0], u_n[2]*u_n[0]))
Qn_old =   as_vector((u_n_old[0], u_n_old[1]*u_n_old[0], u_n_old[2]*u_n_old[0] )) 


#g is gravitational constant
g=9.81


#Flux tensor from SWE
Fu = as_tensor([[h*ux,h*uy], 
				[h*ux*ux+ 0.5*g*h*h-0.5*g*h_b*h_b, h*ux*uy],
				[h*ux*uy,h*uy*uy+0.5*g*h*h-0.5*g*h_b*h_b]
				])

#Flux tensor for SWE if normal flow is 0
Fu_wall = as_tensor([[0,0], 
					[0.5*g*h*h-0.5*g*h_b*h_b, 0],
					[0,0.5*g*h*h-0.5*g*h_b*h_b]
					])


#RHS source vector for SWE is gravity + bottom friction
#can add in things like wind and pressure later
g_vec = as_vector((0,
 					-g*(h-h_b)*h_b.dx(0),
 					-g*(h-h_b)*h_b.dx(1)))
#there are many friction laws, here is an example of a quadratic law
#Linear friction law or quadratic
eps=1e-8
cf=0.00025
mag_v = conditional(pow(ux*ux + uy*uy, 0.5) < eps, 0, pow(ux*ux + uy*uy, 0.5))
fric_vec=as_vector((0,
                    ux*cf*mag_v,
                    uy*cf*mag_v))

S = g_vec+fric_vec



#normal vector
n = FacetNormal(domain)




#begin constructing the weak form, this is standard weak form from IBP
#start adding to residual, beggining with body term
F = -inner(Fu,grad(p))*dx
#add RHS forcing
F += inner(S,p)*dx


#now adding in global boundary terms
#now adding in global boundary terms
for marker, func in boundaries:
	if (marker == 1) or (marker == 2):
		#This is the open boundary in this case
		F += dot(dot(Fu, n), p) * ds(marker)
	else:
		print("Adding wall condition \n\n")
		#this is the wall condition, no flux on this part
		F += dot(dot(Fu_wall, n), p)*ds(marker)

#now adding interior boundary terms using Lax-Friedrichs upwinding for DG
eps=1e-8
#attempt at full expression from https://docu.ngsolve.org/v6.2.1810/i-tutorials/unit-3.4-simplehyp/shallow2D.html
vela =  as_vector((u[1]('+'),u[2]('+')))
velb =  as_vector((u[1]('-'),u[2]('-')))
vnorma = conditional(sqrt(dot(vela,vela)) > eps,sqrt(dot(vela,vela)),eps)
vnormb = conditional(sqrt(dot(velb,velb)) > eps,sqrt(dot(velb,velb)),eps)
C = conditional( (vnorma + sqrt(g*u[0]('+'))) > (vnormb + sqrt(g*u[0]('-'))), (vnorma + sqrt(g*u[0]('+'))) ,  (vnormb + sqrt(g*u[0]('-')))) 
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


vel_plot = fe.Function(V_vel)
vel_plot.name = "depth averaged velocity (m/s)"


#080
xmf=None
results_folder = Path(filename)
results_folder.mkdir(exist_ok=True, parents=True)
wse_writer = io.VTXWriter(domain.comm, results_folder / "WSE.bp", eta_plot, engine="BP4")
vel_writer = io.VTXWriter(domain.comm, results_folder / "vel.bp", vel_plot, engine="BP4")
writers=[wse_writer,vel_writer]

def plot_global_output(u,h_b,V_scalar,V_vel,t,xdmf=None,vtx_writers=None):
	#interpolate and plot water surface elevation and velocity
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
Newton_Solver = CustomNewtonProblem(F,u,dirichlet_conditions, domain.comm, solver_parameters=params,lines=lines,mesh=domain)



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
	u_ex.sub(1).x.array[vel_dirichlet_dofs]=vel_boundary
	u.x.array[h_dirichlet_dofs] = u_ex.x.array[h_dirichlet_dofs]
	u.x.array[vel_dirichlet_dofs] = u_ex.x.array[vel_dirichlet_dofs]
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
	u_ex.sub(1).x.array[vel_dirichlet_dofs]=vel_boundary
	u.x.array[h_dirichlet_dofs] = u_ex.x.array[h_dirichlet_dofs]
	u.x.array[vel_dirichlet_dofs] = u_ex.x.array[vel_dirichlet_dofs]
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

