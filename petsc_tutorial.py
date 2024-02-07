from petsc4py import PETSc
import numpy as np


#give size of matrix

n = 4

#give sparsity pattern
#this means 3 nonzeros allowed per row
nnz = 4 * np.ones(n, dtype=np.int32)
#2 on top and bottom rows
#nnz[0] = nnz[-1] = 2


#initialize matrix
A = PETSc.Mat()
A.createAIJ([n, n], nnz=nnz)

#fill matrix
# First set the first row
#A.setValue(0, 0, 2)
#A.setValue(0, 1, -1)
# Now we fill the last row
#A.setValue(n-1, n-2, -1)
#A.setValue(n-1, n-1, 2)


# And now everything else
a=1
for index in range(n):
	for col in range(n):
		A.setValue(index, col,a)
		a+=1



A.assemble()

#note, attempting to set value outside of where they have already been assigned will throw error
#A.setValue(0, index, -1)
#we can set at entries that are already entered though
#A.setValue(index, index, -1)
#but then need to assemble again
A.assemble()


#must come after assembly
indexptr, indices, data = A.getValuesCSR()
print(indexptr,indices,data)

#this is total on all processes
A.size
#this is local to this process
A.local_size
#see if symmetric
A.isSymmetric()
#get info
print(A.getInfo())

#see if we can view matrix
viewer_obj = PETSc.Viewer()
viewer_obj.createASCII("mat.txt")
viewer_obj.pushFormat(9)
viewer_obj.view(obj=A)

#let's verify permutations
Perm = PETSc.IS()
Perm2 = PETSc.IS()

list1=[0,2,1,3]
list1=np.array(list1,dtype=np.int32)


Perm.createBlock(1,list1)

B=A.permute(Perm,Perm)

#mat permuted
viewer_obj = PETSc.Viewer()
viewer_obj.createASCII("mat_permuted.txt")
viewer_obj.pushFormat(9)
viewer_obj.view(obj=B)

#reverse permute
list2=np.argsort(list1).astype(np.int32)
Perm2.createBlock(1,list2)
C=B.permute(Perm2,Perm2)

#mat reverse permuted
viewer_obj = PETSc.Viewer()
viewer_obj.createASCII("mat_unpermuted.txt")
viewer_obj.pushFormat(9)
viewer_obj.view(obj=C)





