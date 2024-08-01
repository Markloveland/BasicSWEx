from dolfinx import geometry
import numpy as np
from dolfinx import __version__
from ufl import (as_vector,conditional,sqrt,dot,avg,jump,as_tensor)
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


def get_LF_flux(Fu,u,n,eps=1e-8,g=9.81):
    #now adding interior boundary terms using Lax-Friedrichs upwinding for DG
    #attempt at full expression from https://docu.ngsolve.org/v6.2.1810/i-tutorials/unit-3.4-simplehyp/shallow2D.html
    vela =  as_vector((u[1]('+'),u[2]('+')))
    velb =  as_vector((u[1]('-'),u[2]('-')))
    vnorma = conditional(sqrt(dot(vela,vela)) > eps,sqrt(dot(vela,vela)),eps)
    vnormb = conditional(sqrt(dot(velb,velb)) > eps,sqrt(dot(velb,velb)),eps)
    C = conditional( (vnorma + sqrt(g*u[0]('+'))) > (vnormb + sqrt(g*u[0]('-'))), (vnorma + sqrt(g*u[0]('+'))) ,  (vnormb + sqrt(g*u[0]('-')))) 
    Q = as_vector((u[0],u[0]*u[1],u[0]*u[2]))
    flux = dot(avg(Fu), n('+')) + 0.5*C*jump(Q)
    return flux