# Shape matching algorithms: ICP and Optimal Transport based ICP

Two algorithms to solve shape matching problem:

- The fist one is inspired from ICP but also takes scaling into account. The ICP based is highly inspired from https://github.com/ClayFlannigan/icp.

- The second one computes a matching by replacing the first step of the ICP (nearest neighbor matching) by optimal transport (solving linear sum assignment problem with https://github.com/gatagat/lap). This matching algorithm, slower, is theoritecaly the optimal one in term of accuracy.


Point cloud folder is a folder containing some 2D datasets of points clouds to match, for tests purposes.   

In order to test the algorithms on one of these datasets, juste go in "test.py" and  give the variable "objet" a value in {"star","3d","shape"}.
