from dolfinx import geometry
import numpy as np
from dolfinx import __version__

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
def record_stations(u_sol,h_b,f_wd,points_on_proc,cells):
    #saves time series at stations into a numpy array
    #u_values = u_sol.eval(points_on_proc, self.cells)
    #fix for mixed
    h_values = u_sol.sub(0).eval(points_on_proc, cells) - 2*h_b.eval(points_on_proc, cells) + f_wd.eval(points_on_proc,cells)
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