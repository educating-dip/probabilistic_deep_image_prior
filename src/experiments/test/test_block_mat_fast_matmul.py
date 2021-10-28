import torch
import numpy as np


##### SLOW METHOD: #####

# A is a 3*6 random matrix
A = torch.zeros((3,6)).normal_(0, 1)

# B is a 6*6 block-diagonal matrix with blocks of 2*2
B = torch.zeros((6,6))
B[0:2,0:2] = torch.from_numpy(np.array([[1,2],[3,4]]))
B[2:4,2:4] = torch.from_numpy(np.array([[5,6],[7,8]]))
B[4:6,4:6] = torch.from_numpy(np.array([[9,10],[11,12]]))

# multiply A and B to obtain C
C = A @ B

##### FAST METHOD: #####

# divide A into many smaller matrices by slicing vertically;
# each smaller matrix has the same number of coumns as the blocks in B,
# and the same number of rows as the original A
A_reshaped = A.view(3,3,2).permute(1,0,2)


# store B as a series of small matrices, where each small matrix is a block
# from the original B
B_reshaped = torch.zeros((3,2,2))
B_reshaped[0] = torch.from_numpy(np.array([[1,2],[3,4]]))
B_reshaped[1] = torch.from_numpy(np.array([[5,6],[7,8]]))
B_reshaped[2] = torch.from_numpy(np.array([[9,10],[11,12]]))

# matrix multiplication; compare with the original C
C_reshaped = A_reshaped @ B_reshaped
C_reshaped = C_reshaped.permute(0, 2, 1).reshape([6, 3]).t()
import pdb; pdb.set_trace()
