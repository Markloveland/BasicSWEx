import dolfinx.fem.petsc as petsc
from dolfinx import mesh, fem, io
from mpi4py import MPI
import ufl
from ufl import inner, grad, dot, div
import numpy as np
from petsc4py import PETSc
from dolfinx.cpp.mesh import cell_num_entities
from utils import norm_L2, compute_cell_boundary_facets, compute_interior_facet_integration_entities, get_interior_facets
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block



# Create a mesh
comm = MPI.COMM_WORLD
n = 4
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
print("local facet numbering",facets)
# NOTE Despite all facets being present in the submesh, the entity map isn't
# necessarily the identity in parallel
facet_mesh, facet_mesh_to_msh = mesh.create_submesh(msh, fdim, facets)[0:2]
print("facet_mesh_to_msh",facet_mesh_to_msh)
msh_to_facet_mesh = np.full(num_facets, -1)
msh_to_facet_mesh[facet_mesh_to_msh] = np.arange(len(facet_mesh_to_msh))
print("msh to facet mesh",msh_to_facet_mesh)


print("properties of facet mesh")
print("Topology dimension",facet_mesh.topology.dim)
#no map exists in submesh for cells, only knows facets
#facet_mesh.topology.create_entities(1)
#facet_mesh.topology.create_entities(2)
#facet_mesh.topology.create_connectivity(tdim, fdim)
#facet_mesh.topology.create_connectivity(fdim, tdim)
#c_to_f = msh.topology.connectivity(tdim, fdim)
#f_to_c = msh.topology.connectivity(fdim, tdim)
#exit(0)

#maybe this is what needs to change
entity_maps = {facet_mesh: msh_to_facet_mesh}
print("entity maps",entity_maps)


# Create functions spaces
k = 1  # Polynomial degree
V = fem.functionspace(msh, ("Discontinuous Lagrange", k-1))
Vbar = fem.functionspace(facet_mesh, ("Discontinuous Lagrange", k-1))
print(Vbar.tabulate_dof_coordinates())
# Create trial and test functions
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
f = fem.Function(V)
f.interpolate(lambda x: 1*(x[0]<0.51))
ubar, vbar = ufl.TrialFunction(Vbar), ufl.TestFunction(Vbar)

# Create integration entities and define integration measures. We want
# to integrate around each element boundary, so we call the following
# convenience function:
cell_boundary_facets = compute_cell_boundary_facets(msh)



cell_imap = msh.topology.index_map(tdim)
num_cells = cell_imap.size_local + cell_imap.num_ghosts
#quadruplets usually used for dS
interior_facets = compute_interior_facet_integration_entities(
    msh, np.arange(num_cells)
)
print("Interior facets quadruplets",interior_facets)
print(interior_facets)
#plus cells are first two out of every 4
facet_locs,interior_facets_plus, interior_facets_minus  = get_interior_facets(msh,np.arange(num_cells))
#compute by hand and send to submesh
print("Cell boundary facets",cell_boundary_facets)
print("length of cell boundary facets",len(cell_boundary_facets))
print("length of interior boundary facets",len(interior_facets))


print("Numbers of interior facets",facet_locs)


dS = ufl.Measure(
    "ds",
    domain=msh,
    subdomain_data=[
        (0, interior_facets_plus),
        (1, interior_facets_minus)
    ],
)
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
n = ufl.FacetNormal(msh)
L = f*vbar*dS(0) - f*vbar*dS(1)

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
