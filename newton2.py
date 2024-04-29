from dolfinx import fem as fe, nls, log,geometry,io,cpp
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import numpy.linalg as la
import time
from scipy.sparse import csr_matrix
from newton import LinePreconditioner, petsc_to_csr
import matplotlib.pyplot as plt
g=9.81



class CustomNewtonProblem2:

    """An all-in-one class that solves a nonlinear problem. . .
    """
    
    def __init__(self,Fu,Fu_wall,S,dQdt,u,p,n,ds,boundaries,bc,comm,solver_parameters={},mesh=None,lines=None,Fpre=None):
        """initialize the problem
        
        Here we linearize more manually
        F -- Ufl weak form without the flux
        u -- solution vector
        p -- the test functions
        bc -- list of fenicsx dirichlet bcs
        comm -- MPI communicatior
        """

        #the solution vector
        self.u = u
        #the increment variable
        self.u_tilde = ufl.TrialFunction(self.u._V)
        #aliasing to make reading easier
        h, ux, uy = self.u[0], self.u[1], self.u[2]
        #the increment variable
        h_dx, ux_dx, uy_dx = self.u_tilde[0],self.u_tilde[1],self.u_tilde[2]
        self.Fu = Fu
        self.Fu_wall = Fu_wall

        self.p = p

        
        #construct bilinear form directly


        #first, create nonlinear residual

        #now adding interior boundary terms using Lax-Friedrichs upwinding for DG
        eps=1e-8
        #attempt at full expression from https://docu.ngsolve.org/v6.2.1810/i-tutorials/unit-3.4-simplehyp/shallow2D.html
        vela =  ufl.as_vector((self.u[1]('+'),self.u[2]('+')))
        velb =  ufl.as_vector((self.u[1]('-'),self.u[2]('-')))
        vnorma = ufl.conditional(ufl.sqrt(ufl.dot(vela,vela)) > eps,ufl.sqrt(ufl.dot(vela,vela)),eps)
        vnormb = ufl.conditional(ufl.sqrt(ufl.dot(velb,velb)) > eps,ufl.sqrt(ufl.dot(velb,velb)),eps)
        C = ufl.conditional( (vnorma + ufl.sqrt(g*self.u[0]('+'))) > (vnormb + ufl.sqrt(g*self.u[0]('-'))), (vnorma + ufl.sqrt(g*self.u[0]('+'))) ,  (vnormb + ufl.sqrt(g*self.u[0]('-'))))
        flux1 = ufl.dot(ufl.avg(self.Fu), n('+')) 
        flux2 = 0.5*C*ufl.jump(ufl.as_vector((h, h*ux, h*uy )))
        
        temporal_term = ufl.inner(dQdt,p)*ufl.dx
        flux_term1= ufl.inner(flux1, ufl.jump(p))*ufl.dS
        flux_term2 = ufl.inner(flux2, ufl.jump(p))*ufl.dS
        body_term = ufl.inner(Fu,ufl.grad(p))*ufl.dx
        source_term = ufl.inner(S,p)*ufl.dx


        
        
        #now adding in global boundary terms
        i = 0
        for marker, func in boundaries:
            if (marker == 1) or (marker == 2):
                if i==0:
                    boundary_term = ufl.dot(ufl.dot(self.Fu, n), self.p) * ds(marker)
                    i+=1
                else:
                    boundary_term+=ufl.dot(ufl.dot(self.Fu, n), self.p) * ds(marker)
            else:
                if i==0:
                    boundary_term = ufl.dot(ufl.dot(self.Fu_wall, n), self.p)*ds(marker)
                    i+=1
                else:
                    boundary_term += ufl.dot(ufl.dot(self.Fu_wall, n), self.p)*ds(marker)


        self.F = temporal_term + flux_term1 + flux_term2  - body_term + source_term + boundary_term
        self.residual = fe.form(self.F)
        
        #now create bilinear form that will be the matrix
        #everything will be same except second flux term
        self.J = ufl.derivative(temporal_term + flux_term1 - body_term + source_term + boundary_term , self.u, self.u_tilde)
        #add linearized flux2 term manually
        self.J+=ufl.inner(0.5*C*ufl.jump(ufl.as_vector((h_dx, h_dx*ux + h*ux_dx, h_dx*uy + h*uy_dx ))), ufl.jump(p))*ufl.dS
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