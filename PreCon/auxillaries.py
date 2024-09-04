from dolfinx import geometry
import numpy as np
from dolfinx import __version__
from ufl import (as_vector,conditional,sqrt,dot,avg,jump,as_tensor)
from dolfinx.cpp.mesh import cell_num_entities
def init_stations(domain,points):
    #reads in recording stations and outputs points on each processor
    try:
        #060 old version
        bb_tree = geometry.BoundingBoxTree(domain, domain.topology.dim)
    except:
        #080 later versions
        bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    try:
        #060
        cell_candidates = geometry.compute_collisions(bb_tree, points)
    except:
        #080
        cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    for i, point in enumerate(points):
        if len(colliding_cells.links(i))>0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    return cells,np.array(points_on_proc, dtype=np.float64)
def record_stations(u_sol,points_on_proc,cells):
    #saves time series at stations into a numpy array
    #u_values = u_sol.eval(points_on_proc, self.cells)
    #fix for mixed
    h_values = u_sol.sub(0).eval(points_on_proc, cells)
    u_values = u_sol.sub(1).eval(points_on_proc, cells)
    u_values = np.hstack([h_values,u_values])
    return u_values

def gather_stations(root,comm,local_stats,local_vals):
    rank = comm.Get_rank()
    #PETSc.Sys.Print("Trying to mpi gather")
    gathered_coords = comm.gather(local_stats,root=root)
    gathered_vals = comm.gather(local_vals,root=root)
    coords=[]
    vals = []
    if rank == root:
        for a in gathered_coords:
            if a.shape[0] != 0:
                for row in a:
                    coords.append(row)
        coords = np.array(coords)
        coords,ind1 = np.unique(coords,axis=0,return_index=True)
        for n in gathered_vals:
            if n.shape[1] !=0:
                for row in n:
                    vals.append(np.array(row))
        vals = np.array(vals)
    return coords,vals

def get_F(u,h_b,g=9.81):
    h, ux, uy = u[0], u[1], u[2]
    return as_tensor([[h*ux,h*uy], 
                [h*ux*ux+ 0.5*g*h*h-0.5*g*h_b*h_b, h*ux*uy],
                [h*ux*uy,h*uy*uy+0.5*g*h*h-0.5*g*h_b*h_b]
                ])

def get_LF_flux_form(Fu,u,n,vbar,dSplus,dSminus,dSext,eq_no,eps=1e-8,g=9.81):
    #now adding interior boundary terms using Lax-Friedrichs upwinding for DG
    #attempt at full expression from https://docu.ngsolve.org/v6.2.1810/i-tutorials/unit-3.4-simplehyp/shallow2D.html
    vel =  as_vector((u[1],u[2]))
    vnorm = conditional(sqrt(dot(vel,vel)) > eps,sqrt(dot(vel,vel)),eps)
    C = vnorm + sqrt(g*u[0])
    Q = as_vector((u[0],u[0]*u[1],u[0]*u[2]))
    T1 = dot(Fu, n)
    T2 = 0.5*C*Q
    #just L1 norm for now
    #because normals cancel must flip sign
    return 0.5*T1[eq_no]*vbar*dSplus - 0.5*T1[eq_no]*vbar*dSminus + T2[eq_no]*vbar*dSplus - T2[eq_no]*vbar*dSminus +T1[eq_no]*vbar*dSext

def compute_cell_boundary_facets(msh):
    """Compute the integration entities for integrals around the
    boundaries of all cells in msh.

    Parameters:
        msh: The mesh.

    Returns:
        Facets to integrate over, identified by ``(cell, local facet
        index)`` pairs.
    """
    tdim = msh.topology.dim
    fdim = tdim - 1
    n_f = cell_num_entities(msh.topology.cell_type, fdim)
    n_c = msh.topology.index_map(tdim).size_local
    return np.vstack(
        (np.repeat(np.arange(n_c), n_f), np.tile(np.arange(n_f), n_c))
    ).T.flatten()

def compute_interior_facet_integration_entities(msh, cell_map):
    """
    Compute the integration entities for interior facet integrals.

    Parameters:
        msh: The mesh
        cell_map: A map to apply to the cells in the integration entities

    Returns:
        A (flattened) list of pairs of (cell, local facet index) pairs
    """
    # FIXME Do this more efficiently
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_entities(fdim)
    msh.topology.create_connectivity(tdim, fdim)
    msh.topology.create_connectivity(fdim, tdim)
    c_to_f = msh.topology.connectivity(tdim, fdim)
    f_to_c = msh.topology.connectivity(fdim, tdim)
    integration_entities = []
    for facet in range(msh.topology.index_map(fdim).size_local):
        cells = f_to_c.links(facet)
        if len(cells) == 2:
            # FIXME Don't use tolist
            local_facet_plus = c_to_f.links(cells[0]).tolist().index(facet)
            local_facet_minus = c_to_f.links(cells[1]).tolist().index(facet)

            integration_entities.extend(
                [
                    cell_map[cells[0]],
                    local_facet_plus,
                    cell_map[cells[1]],
                    local_facet_minus,
                ]
            )
    return integration_entities


def get_interior_facets(msh,cell_map):
    """
    Compute the integration entities for interior facet integrals.

    Parameters:
        msh: The mesh
        cell_map: A map to apply to the cells in the integration entities

    Returns:
        A (flattened) list of pairs of (cell, local facet index) pairs
    """
    # FIXME Do this more efficiently
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_entities(fdim)
    msh.topology.create_connectivity(tdim, fdim)
    msh.topology.create_connectivity(fdim, tdim)
    c_to_f = msh.topology.connectivity(tdim, fdim)
    f_to_c = msh.topology.connectivity(fdim, tdim)
    facet_plus = []
    facet_minus = []
    facet_boundary = []
    facets = []
    for facet in range(msh.topology.index_map(fdim).size_local):
        cells = f_to_c.links(facet)
        if len(cells) == 2:
            # FIXME Don't use tolist
            local_facet_plus = c_to_f.links(cells[0]).tolist().index(facet)
            local_facet_minus = c_to_f.links(cells[1]).tolist().index(facet)

            facet_plus.extend(
                [
                    cell_map[cells[0]],
                    local_facet_plus
                ]
            )
            facet_minus.extend(
                [
                    cell_map[cells[1]],
                    local_facet_minus,
                ]
                )
            facets.extend(
                [
                    facet
                ]
                )
        else:
            local_facet_plus = c_to_f.links(cells[0]).tolist().index(facet)
            facet_boundary.extend(
                [
                    cell_map[cells[0]],
                    local_facet_plus,
                ]
                )


    return facets,facet_plus, facet_minus, facet_boundary


#compute the norm between the arrays
def compute_norm_special(x_con,x_momx,x_momy,ord):
    #compute norm elementwise over 3 numpy vectors
    X = np.column_stack((x_con,x_momx,x_momy))
    #take norm
    return np.linalg.norm(X,ord=ord,axis=1)

