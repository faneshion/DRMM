

#wget http://

#EIGEN_SRC="-I/home/fyx/.local/include/eigen3/ -I/home/fyx/.local/include -L/home/fyx/.local/lib"
g++ main.cpp -o ./NN4IR NN4IR.cpp util.cpp Config.cpp -std=c++11  -O3 -w -funroll-loops -fopenmp #-g

