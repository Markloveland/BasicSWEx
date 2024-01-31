from dolfinx import fem as fe, nls, log,geometry,io,cpp
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import numpy.linalg as la
import time
from scipy.sparse import csr_matrix

def petsc_to_csr(A):
    indptr, indices, data = A.getValuesCSR()
    return csr_matrix((data, indices, indptr), shape=A.size)

class CustomNewtonProblem:

    """An all-in-one class that solves a nonlinear problem. . .
    """
    
    def __init__(self,F,u,bc,comm,solver_parameters={},mesh=None):
        """initialize the problem
        
        F -- Ufl weak form
        u -- solution vector
        bc -- list of fenicsx dirichlet bcs
        comm -- MPI communicatior
        """
        self.u = u
        self.F = F
        self.residual = fe.form(self.F)

        self.J = ufl.derivative(self.F, self.u)
        self.jacobian = fe.form(self.J)

        self.bcs = bc
        self.comm = comm
        #relative tolerance for Newton solver
        self.rtol = 1e-5
        #absolute tolerance for Newton solver
        self.atol = 1e-6
        #max iteration number for Newton solver
        self.max_it = 5
        #relaxation parameter for Newton solver
        self.relaxation_parameter = 1.00
        #underlying linear solver
        #default for serial is lu, default for mulitprocessor is gmres
        if self.comm.Get_size() == 1:
            print("serial run")
            self.ksp_type = "gmres"#preonly
            self.pc_type = "ilu"#lu
        else:
            self.ksp_type = "gmres"
            self.pc_type = "bjacobi"

        #overwrite any proprties by passing the dictionary called solver_parameters
        for k, v in solver_parameters.items():
            setattr(self, k, v)


        self.A = fe.petsc.create_matrix(self.jacobian)
        self.L = fe.petsc.create_vector(self.residual)
        self.solver = PETSc.KSP().create(self.comm)

        self.solver.setTolerances(rtol=1e-8, atol=1e-9, max_it=1000)
        self.solver.setOperators(self.A)
        if self.pc_type == 'line_smooth':
            self.pc = LinePreconditioner(self.A, mesh)
        else:
            self.pc = self.solver.getPC()
            self.pc.setType(self.pc_type)


    def log(self, *msg):
        if self.comm.rank == 0:
            print(*msg)
    
    def solve(self, u, max_it=5):
        """Solve the nonlinear problem at u
        """

        dx = fe.Function(u._V)
        i = 0
        rank = self.comm.rank
        A, L, solver = self.A, self.L, self.solver

        while i < self.max_it:
            # Assemble Jacobian and residual
            with L.localForm() as loc_L:
                loc_L.set(0)
            A.zeroEntries()
            fe.petsc.assemble_matrix(A, self.jacobian, bcs=self.bcs)
            A.assemble()
            fe.petsc.assemble_vector(L, self.residual)
            L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            L.scale(-1)
            # Compute b - J(u_D-u_(i-1))
            fe.petsc.apply_lifting(L, [self.jacobian], [self.bcs], x0=[u.vector], scale=1)
            # Set dx|_bc = u_{i-1}-u_D
            fe.petsc.set_bc(L, self.bcs, u.vector, 1.0)
            L.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

            # Solve linear problem
            if self.pc_type == 'line_smooth':
                #print("A cond num", la.cond(petsc_to_csr(A).todense()))
                new_A, new_rhs = self.pc.precondition(L)
                #print("new_A cond num", la.cond(petsc_to_csr(new_A).todense()))
                #solver.reset()
                #print(solver.getPC().getType())
                #raise ValueError()
                solver = PETSc.KSP().create(self.comm)
                solver.setType("gmres")
                solver.setTolerances(rtol=1e-8, atol=1e-9)
                solver.getPC().setType("mat")
                solver.setOperators(A, self.pc.mat)
                start = time.time()
                solver.solve(L, dx.vector)
                print("solved in ", time.time()-start)
            else:
                start = time.time()
                #print("pc type", solver.getPC().getType())
                #print("A cond num", la.cond(petsc_to_csr(A).todense()))
                solver.solve(L, dx.vector)

                print("solved in ", time.time()-start)
            
            dx.x.scatter_forward()
            self.log(f"linear solver convergence {solver.getConvergedReason()}" +
                    f", iterations {solver.getIterationNumber()}")
            # Update u_{i+1} = u_i + delta x_i
            #not working in parallel?
            u.x.array[:] += self.relaxation_parameter*dx.x.array[:]
            
            i += 1
            
            if i == 1:
                self.dx_0_norm = dx.vector.norm(0)
                self.log('dx_0 norm,',self.dx_0_norm)
            #print('dx before', dx.vector.getArray())
            if self.dx_0_norm > 1e-8:
                dx.x.array[:] = np.array(dx.x.array[:]/self.dx_0_norm)

            dx.vector.assemble()
            
            # Compute norm of update
            correction_norm = dx.vector.norm(0)

         
            self.log(f"Netwon Iteration {i}: Correction norm {correction_norm}")
            if correction_norm < self.atol:
                break
class LinePreconditioner:

    def __init__(self, A, mesh):
        """Initialize the preconditioner from the 
        """

        dim = mesh.topology.dim
        num_cells = mesh.topology.index_map(dim).size_local
        print("local num cells", num_cells)
        #hardcoded for now
        block_size = 2*A.size[0] // num_cells
        print("block size", block_size)
        print("Matri size", A.size)
        self.block_size = block_size
        self.A = A
        mat = PETSc.Mat()
        #set original sparsity pattern
        #will be equal to number of dof per line
        mat.createAIJ((A.size[0], A.size[0]), nnz=np.full(A.size[0], block_size, dtype=np.int32), comm=mesh.comm)
        mat.setUp()
        mat.setBlockSize(block_size)
        self.mat = mat

    def precondition(self, rhs):
        """Apply the block preconditioner to A and the right hand side

        returns P^-1 * A, P^-1 * rhs
        """

        old_block_size = self.A.getBlockSize()
        self.A.setBlockSize(self.block_size)
        inv = self.A.invertBlockDiagonal()
        self.A.setBlockSize(old_block_size)
        start_ind, stop_ind = self.mat.owner_range
        block_inds = np.arange(start_ind//self.block_size, stop_ind//self.block_size+1)
        block_inds = block_inds.astype(np.int32)
        self.mat.setValuesBlockedCSR(block_inds, block_inds[:-1], inv)
        #for i in range(len(inv)):
        #    mat.setValuesBlocked(block_inds[i:i+1], block_inds[i:i+1], inv[i])
        self.mat.assemble()
        new_rhs = self.mat.createVecRight()
        self.mat.mult(rhs, new_rhs)
        return self.mat.matMult(self.A), new_rhs

