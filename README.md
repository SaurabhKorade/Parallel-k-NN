# Parallel-k-NN
Title:
	A parallel k-NN program that finds the k nearest neighbors of a given query point. 
	
Description:	
	The parallel k-NN program is implemented using kd-trees and the generation of kd-tree is parallelized using POSIX threads. The tree is balanced by sorting the data points in the appropriate dimension and choosing the median as the node starting from root. Euclidean distance is calculated between query point and node of the tree and the tree is traversed according to the value of the dimension of the query point. The dimensions alternate in a round-robin fashion as the depth increases. For instance, consider query point (1,2,3) and kd-node point (5,4,3) for dimension 0(Zeroeth level of tree), as 1 < 5 the tree is searched for the left child for nearest neighbors.
	
Input and Output:
	The training point and data points are read from binary files and the filenames are passed as arguments to the program. The result for all the neighbors of the query points are dumped in a binary file with the filename provided as argument. Total time for creation of the kd-tree and exectuion of queries is printed in seconds.
	
Compiling:
	Complie the c++ code using the Makefile provided. 
	
Execution:
	he program take four arguments on the command line ./k-nn n_cores training_file query_file result_file
	Example of run command: srun -n 1 -c 20 ./knn 13 data_2004601.dat query_2003955.dat result1
	or: ./knn 20 data_2004601.dat query_2003955.dat result2
	
Author:
	Name - Saurabh Korade
	BU-ID - B00813358
	skorade1@binghamton.edu
