from dolfinx import fem as fe, nls, log,geometry,io,cpp
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import numpy.linalg as la
import time
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

def petsc_to_csr(A):
    indptr, indices, data = A.getValuesCSR()
    return csr_matrix((data, indices, indptr), shape=A.size)

class CustomNewtonProblem:

    """An all-in-one class that solves a nonlinear problem. . .
    """
    
    def __init__(self,F,u,bc,comm,solver_parameters={},mesh=None,lines=None,Fpre=None):
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
            #added for line pre
            self.F_pre=Fpre
            self.residual_pre = fe.form(self.F_pre)
            self.J_pre = ufl.derivative(self.F_pre, self.u)
            self.jacobian_pre = fe.form(self.J_pre)

            self.A_pre = fe.petsc.create_matrix(self.jacobian_pre)
            self.L_pre = fe.petsc.create_vector(self.residual_pre)




            self.pc = LinePreconditioner(self.A_pre, mesh,lines)
        else:
            self.pc = self.solver.getPC()
            self.pc.setType(self.pc_type)
            dim = mesh.topology.dim
            num_cells = mesh.topology.index_map(dim).size_local
            self.block_size = self.A.size[0] // num_cells


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
        if self.pc_type == 'line_smooth':
            A_pre,L_pre = self.A_pre,self.L_pre

        while i < self.max_it:
            # Assemble Jacobian and residual
            with L.localForm() as loc_L:
                loc_L.set(0)
            A.zeroEntries()
            fe.petsc.assemble_matrix(A, self.jacobian, bcs=self.bcs)
            A.assemble()
            plt.spy(petsc_to_csr(A),markersize=1)
            plt.savefig("main_mat.png")
            plt.close()
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
                #for performance don't compute this
                #new_A, new_rhs = self.pc.precondition(L)
                A_pre.zeroEntries()
                fe.petsc.assemble_matrix(A_pre, self.jacobian_pre, bcs=self.bcs)
                A_pre.assemble()
                self.pc.precondition(L)
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
                #unpermute to avoid issues changing matrix
                self.pc.mat=self.pc.mat.permute(self.pc.Perm,self.pc.Perm)
                print("solved in ", time.time()-start)
            else:
                start = time.time()

                #print("pc type", solver.getPC().getType())
                #print("A cond num", la.cond(petsc_to_csr(A).todense()))
                #A.setBlockSize(self.block_size)
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

    def __init__(self, A, mesh,lines):
        """Initialize the preconditioner 
        """
        self.mesh=mesh
        dim = mesh.topology.dim
        num_cells = mesh.topology.index_map(dim).size_local
        print("local num cells", num_cells)
        #hardcoded for now
        #a block size is dofs per cell
        self.block_size = A.size[0] // num_cells

        
        print("element block size", self.block_size)
        
        print("Matri size", A.size)
        #hard code for now, will think about more later
        self.line_block_size = lines.shape[1]*self.block_size
        
        print("line block size",self.line_block_size)
        self.A = A
        mat = PETSc.Mat()
        #set original sparsity pattern
        #will be equal to number of dof per line

        #we also know that the preconditioner matrix will be at most block tridiag
        mat.createAIJ((A.size[0], A.size[0]), nnz=np.full(A.size[0], self.line_block_size, dtype=np.int32), comm=mesh.comm)
        mat.setUp()
        mat.setBlockSize(self.line_block_size)
        self.mat = mat

        #we also build a permutation matrix

        self.Perm = PETSc.IS()
        self.Perm2 = PETSc.IS()
        #take in indeces from line by appending all the lines contiguously
        #list1=[]
        #for a in lines:
        #list1+=a
        list1 = lines.flatten()
        
        list1=np.array(list1,dtype=np.int32)
        list2=np.argsort(list1).astype(np.int32)
        print("List of lines", list1)
        print("Transpose", list2)
        #this is the forward permutation on LHS
        self.Perm.createBlock(self.block_size,list1,comm=mesh.comm)

        #this is the inverse permutation on RHS
        self.Perm2.createBlock(self.block_size,list2,comm=mesh.comm)


        #check that it is doing the right thing
        print(self.Perm.getIndices())
        print(self.Perm2.getIndices())




    def precondition(self, rhs):
        """Apply the line smoothing preconditioner to A and the right hand side

        returns P^-1 * A, P^-1 * rhs
        updates pc mat,
        this pc mat will be

        Q'B^-1Q
        """
        old_block_size = self.A.getBlockSize()
        #print("old blocksize",old_block_size)

        #first get A and permute rows and columns
        #B=self.A.duplicate(copy=True)
        #permute but need to change blocksize
      
        plt.spy(petsc_to_csr(self.A),markersize=1)
        plt.savefig("A_before_permutation.png")
        plt.close()
        self.A.setBlockSize(self.block_size)
        B=self.A.permute(self.Perm,self.Perm)
        self.A.setBlockSize(old_block_size)
        plt.spy(petsc_to_csr(B),markersize=1)
        plt.savefig("A_after_permutation.png")
        plt.close()


        #this is now in right order, but need to truncate rows/cols
        #will figure this out later
        #now deal with permuted matrix
        #B.zeroRowsColumns ?
        

        

        #fill in entries for mat
        #eventually we want a block tridiagonal solve efficiently but leave for now
        B.setBlockSize(self.line_block_size)
        inv = B.invertBlockDiagonal()
        #B.setBlockSize(old_block_size)
        start_ind, stop_ind = self.mat.owner_range
        #print(inv.shape,start_ind,stop_ind)
        block_inds = np.arange(start_ind//self.line_block_size, stop_ind//self.line_block_size+1)
        block_inds = block_inds.astype(np.int32)
        #print(block_inds)
        #print(block_inds)
        #print(inv)
        #neeed to rewrite this but works for now
        #mat=PETSc.Mat()
        #mat.createAIJ((self.A.size[0], self.A.size[0]), nnz=np.full(self.A.size[0], self.line_block_size, dtype=np.int32), comm=self.mesh.comm)
        #mat.setUp()
        #mat.setBlockSize(self.line_block_size)
        self.mat.setBlockSize(self.line_block_size)
        self.mat.setValuesBlockedCSR(block_inds, block_inds[:-1], inv)

        self.mat.setBlockSize(self.block_size)
        self.mat.assemble()
        
        #plt.spy(petsc_to_csr(self.mat),markersize=1)
        #plt.savefig("Preconditioner_notpermuted.png")
        #plt.close()

        self.mat=self.mat.permute(self.Perm2,self.Perm2)
        #mat.assemble()
        #self.mat=mat.duplicate(copy=True)
        B.destroy()


        #plt.spy(petsc_to_csr(self.mat),markersize=1)
        #plt.savefig("Preconditioner.png")
        #plt.close()


        #new_rhs = self.mat.createVecRight()
        #self.mat.mult(rhs, new_rhs)
        #return self.mat.matMult(self.A), new_rhs
        return 0

