

#wget http://
#wget http://
#wget http://
#wget http://

#tar jxf *.tar.bz2
#tar jxf *.tar.bz2
#tar jxf *.tar.bz2
#tar jxf *.tar.bz2

g++ main.cpp -o ./NN4IR NN4IR.cpp util.cpp Config.cpp -std=c++11  -O3 -w -funroll-loops -fopenmp 

./NN4IR -config config.ini > NN4IR.log &

tail -f -n 1000 NN4IR.log


## evaluation of the ranklist
#./trec_eval ./rob04-title/qrels.rob04.txt DRMM-LCH-IDF-rob04-title.ranklist -m all_trec

