from dolfinx import fem as fe, nls, log,geometry,io,cpp
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from scipy.sparse import csr_matrix
import numpy.linalg as la
import time


class CustomNewtonProblem:

    """An all-in-one class that solves a nonlinear problem. . .
    """
    
    def __init__(self,F,u,bc,comm,solver_parameters={}):
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
            #start = time.time()
            #print("pc type", solver.getPC().getType())
            solver.solve(L, dx.vector)
            #print("solved in ", time.time()-start)
            
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
