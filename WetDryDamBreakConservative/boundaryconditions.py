from dolfinx import fem as fe
import dolfinx.mesh as mesh
from ufl import (div, as_tensor, as_vector, inner, dx,ds,Measure)
import numpy as np

def BoundaryCondition(type,marker,bound_func,forcing_func,V):
    #a function which identifies d.o.f numbers for given boundary conditions
    #for now only set up for open boundry
    if type == "Open":
        #works for DG
        dofs = fe.locate_dofs_geometrical((V, V.collapse()[0]), bound_func)[0]
        bc = fe.dirichletbc(forcing_func, dofs)
    else:
        dofs = np.array([])
        bc = None
        raise Warning("Unknown boundary condition: {0:s}".format(type))

    return dofs,bc
            


def MarkBoundary(domain,boundaries):

    facet_indices, facet_markers = [], []
    fdim = domain.topology.dim - 1

    #need to restrict to boundary facets, works for now byt not sure if works in general
    #WARNING: Not sure if this works for DG yet, but at least for tidal case it does
    boundary_facets=mesh.locate_entities_boundary(domain,fdim,lambda x:np.full(x.shape[1],True,dtype=bool))

    for (marker, locator) in boundaries:
        facets = mesh.locate_entities(domain, fdim, locator)
        facets = np.intersect1d(facets,boundary_facets)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    facet_tag = mesh.meshtags(domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
    return facet_markers,facet_tag
