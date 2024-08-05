import dolfinx.fem.petsc as petsc
from dolfinx import mesh, fem, io
from mpi4py import MPI
import ufl
from ufl import inner, grad, dot, div
import numpy as np
from petsc4py import PETSc
from dolfinx.cpp.mesh import cell_num_entities
from utils import norm_L2, compute_cell_boundary_facets, compute_interior_facet_integration_entities
from utils import   get_interior_facets, get_F, get_LF_flux_form
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
from basix.ufl import element, mixed_element
from petsc4py.PETSc import ScalarType

# Create a mesh
comm = MPI.COMM_WORLD
n = 1
msh = mesh.create_unit_square(comm, n, n, mesh.CellType.triangle)

# Create a sub-mesh of all facets in the mesh to allow the facet function
# spaces to be created
tdim = msh.topology.dim
fdim = tdim - 1
num_cell_facets = cell_num_entities(msh.topology.cell_type, fdim)
print("Num cell facets = ",num_cell_facets)

#create facet connectivities
msh.topology.create_entities(fdim)
#save facet imap
facet_imap = msh.topology.index_map(fdim)
num_facets = facet_imap.size_local + facet_imap.num_ghosts
facets = np.arange(num_facets, dtype=np.int32)
#print("local facet numbering",facets)
# NOTE Despite all facets being present in the submesh, the entity map isn't
# necessarily the identity in parallel
facet_mesh, facet_mesh_to_msh = mesh.create_submesh(msh, fdim, facets)[0:2]
#print("facet_mesh_to_msh",facet_mesh_to_msh)
msh_to_facet_mesh = np.full(num_facets, -1)
msh_to_facet_mesh[facet_mesh_to_msh] = np.arange(len(facet_mesh_to_msh))
#print("msh to facet mesh",msh_to_facet_mesh)
#use this to map facets on facet mesh to mother mesh
entity_maps = {facet_mesh: msh_to_facet_mesh}
#print("entity maps",entity_maps)
#print("properties of facet mesh")
#print("Topology dimension",facet_mesh.topology.dim)
#no map exists in submesh for cells, only knows facets
#facet_mesh.topology.create_entities(1)
#facet_mesh.topology.create_entities(2)
#facet_mesh.topology.create_connectivity(tdim, fdim)
#facet_mesh.topology.create_connectivity(fdim, tdim)
#c_to_f = msh.topology.connectivity(tdim, fdim)
#f_to_c = msh.topology.connectivity(fdim, tdim)
#exit(0)





# Create functions spaces
k = 1  # Polynomial degree
#regular DG that lives on cells

#print(Vbar.tabulate_dof_coordinates())
# Create trial and test functions


#create LF flux as will be used in code
#'''
el_h   = element("DG", msh.basix_cell(), degree=k-1)
el_vel = element("DG", msh.basix_cell(), degree=k-1, shape = (2,))
V = fem.functionspace(msh, mixed_element([el_h, el_vel]))
u = fem.Function(V)

#get the scalar space for h_b
V_scalar = V.sub(0).collapse()[0]
h_b = fem.Function(V_scalar)
depth = 5
h_b.interpolate(lambda x: depth + 0*x[0])
utrial, v = ufl.TrialFunction(V_scalar), ufl.TestFunction(V_scalar)
#

u_const = 10

#initialize u wirh some crap
u.sub(0).interpolate(
    fem.Expression(
        h_b, 
        V.sub(0).element.interpolation_points()))
  
u.sub(1).interpolate(
    fem.Expression(
        ufl.as_vector([fem.Constant(msh, ScalarType(-u_const)),
            fem.Constant(msh, ScalarType(u_const))]),
        V.sub(1).element.interpolation_points()))
print("u",u.x.array[:])

#'''
V = fem.functionspace(msh, ("Discontinuous Lagrange", k-1))
f=fem.Function(V)
f.interpolate(lambda x: 1*(x[0]<100))

print(V.tabulate_dof_coordinates())

#DG that lives on mesh skeleton
Vbar = fem.functionspace(facet_mesh, ("Discontinuous Lagrange", k-1))
ubar, vbar = ufl.TrialFunction(Vbar), ufl.TestFunction(Vbar)
# Create integration entities and define integration measures. We want
# to integrate around each element boundary, so we call the following
# convenience function:
cell_boundary_facets = compute_cell_boundary_facets(msh)



cell_imap = msh.topology.index_map(tdim)
num_cells = cell_imap.size_local + cell_imap.num_ghosts
#quadruplets usually used for dS
all_interior_facets = compute_interior_facet_integration_entities(
    msh, np.arange(num_cells)
)
#print("Interior facets quadruplets",interior_facets)
#print(interior_facets)
#plus cells are first two out of every 4
facet_locs,interior_facets_plus, interior_facets_minus  = get_interior_facets(msh,np.arange(num_cells))
#compute by hand and send to submesh
#print("Cell boundary facets",cell_boundary_facets)
#print("length of cell boundary facets",len(cell_boundary_facets))
#print("length of interior boundary facets",len(interior_facets))
print("Numbers of interior facets",facet_locs)


dS = ufl.Measure(
    "ds",
    domain=msh,
    subdomain_data=[
        (0, interior_facets_plus),
        (1, interior_facets_minus)
    ],
)
#doesnt seem to work

dS_2 = ufl.Measure(
    "dS",
    domain=msh,
    subdomain_data=[
        (1, all_interior_facets)
    ],
)
ds_c = ufl.Measure("ds", subdomain_data=[(0, cell_boundary_facets)], domain=msh)

# Create a cell integral measure over the facet mesh
dx_f = ufl.Measure("dx", domain=facet_mesh)

#now see if we can construct an integral over each facet
A = ubar*vbar*dx_f
Aform = fem.form(A,entity_maps=entity_maps)
Amat = petsc.assemble_matrix(Aform)
Amat.assemble()
nrow,ncol = Amat.getSize()
print(Amat.getValues(range(nrow),range(ncol)))

#always restrict to positive, instead switch with 0 and +

#L = f*vbar*dS(0) + f*vbar*dS(1)

#only way to do conditional is to integrate and then take max vector
#there is error in C but good enough most likely
n = ufl.FacetNormal(msh)
Fu = get_F(u,h_b)
L = get_LF_flux_form(Fu,u,n,vbar,dS(0),dS(1))
n = ufl.FacetNormal(msh)
sample_flux = ufl.as_vector((-f,f))
L = ufl.dot(sample_flux, n)*vbar*dS(0) # ufl.dot(Fu[0], n)*vbar*dS(1)

#maybe the - restriction is what works for some reason
#L = ufl.avg(f)*vbar('-')*dS_2(0)
Lform = fem.form(L,entity_maps=entity_maps)
Lvec = petsc.assemble_vector(Lform)
Lvec.assemble()
nrow = Lvec.getSize()
print(Lvec.getValues(range(nrow)))

#now solve
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(Amat)
ksp.setType("preonly")
ksp.getPC().setType("lu")
# Compute solution
x = Amat.createVecRight()
#solve
ksp.solve(Lvec, x)
print("L2 interpolation to facets",x.getValues(range(nrow)))

from dolfinx.io import VTXWriter
with VTXWriter(msh.comm, "u.bp", h_b, "bp4") as f:
    f.write(0.0)
