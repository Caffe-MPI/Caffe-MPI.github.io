
Caffe-MPI for Deep Learning
============ 
![ID](http://image.baidu.com/search/down?tn=download&word=download&ie=utf8&fr=detail&url=http%3A%2F%2Fpic27.nipic.com%2F20130401%2F2822139_101059315193_2.jpg&thumburl=http%3A%2F%2Fimg2.imgtn.bdimg.com%2Fit%2Fu%3D38285803%2C1699496659%26fm%3D21%26gp%3D0.jpg)
#Introduction

Caffe-MPI is a deep learning framework designed for both efficiency and flexibility, developed by HPC development team of inspur. It is a GPU cluster version, which is designed and developed on the BVLC single GPU version ( https://github.com/BVLC/caffe, more details please visit http://caffe.berkeleyvision.org).

#Features
 
####(1) The design based on HPC system 
The design of Caffe-MPI is based on HPC system architecture; System hardware: Lustre+IB+GPU; it adopts multi process and multi thread to read the training data in parallel which can be achieve higher IO throughput in this way; the parameters fast transmission and model updating through IB network; The software programming model uses MPI+PThread+CUDA, MPI communication between each node, PThread and CUDA threads parallelism in the node;

####(2) High performance and high scalability
The model can be trained on multi-node-multi-GPU-card platform through Caffe-MPI, we got a better performance improvement compared with BVLC Caffe-master version, Caffe-MPI can be implemented for large-scale data training, the performance of goolgenet we trained through Caffe-MPI is 13 times than the performance trained through BVLC Caffe-master. It supports above 16+ GPUs extension, and the parallel efficiency can reach more than 80%.

###(3) Good inheritance and easy-using
Caffe-MPI retains all the features of the original Caffe architecture, namely the pure C++/CUDA architecture, support of the command line, Python interfaces, and various programming methods. As a result, the cluster version of the Caffe framework is user-friendly, fast, modularized and open, and gives users the optimal application experience. 

#How to use it
  See Caffe-MPI user guide.pdf

#Try your first MPI Caffe
This program can run 2 processes at least.

####cifar10
  1.	Run data/cifar10/get_cifar10.sh to get cifar10 data.
  2.	Run examples/cifar10/create_cifar10.sh to conver raw data to leveldb format.
  3.	Run examples/cifar10/mpi_train_quick.sh to train the net. You can modify the "-n 16" to set new process number where 16 is the number of parallel processes, (if you use GPUs, the process number is m+1, m is GPU number) the "-host node11" is the node name in mpi_train_quick.sh script.
  4.	Example of mpi_train_quick.sh script.
mpirun -machinefile hostsib -n 17 ./build/tools/caffe train \ --solver=examples/cifar10/cifar10_quick_solver.prototxt 

#Reference

  *	More Effective Distributed ML via a Stale Synchronous Parallel Parameter Server

  *	Deep Image: Scaling up Image Recognition

#Ask Questions

  * For reporting bugs, please use the caffe-mpi/issues page or send email to us.

  * Email address: Caffe@inspur.com

#Author

    Zhang,Qing; Wang,Yajuan;Gong;Zhan; Shen,Bo ;

#Acknowledgements

 The Caffe-MPI developers would like to thank QiHoo(Zhang,Gang ; Dr.Hu,Jinhui) Nvidia(Dr.Simon See ; Jessy Huan; joey Wang) for algorithm support and Inspur for guidance during Caffe-MPI development.

