

#wget http://www.bigdatalab.ac.cn/benchmark/upload/download_source/5e67ebd5-6e32-74a9-294f-45c753dce0af_rob04-title.tar.bz2 
#tar jxf 5e67ebd5-6e32-74a9-294f-45c753dce0af_rob04-title.tar.bz2

#wget http://www.bigdatalab.ac.cn/benchmark/upload/download_source/8370623d-58e0-5685-4208-38d5d1cab60f_rob04-desc.tar.bz2
#tar jxf 8370623d-58e0-5685-4208-38d5d1cab60f_rob04-desc.tar.bz2

#wget http://www.bigdatalab.ac.cn/benchmark/upload/download_source/d987ad9f-d852-ccdf-4e26-5c1b294cf81b_clueweb09B-desc.tar.bz2 
#tar jxf d987ad9f-d852-ccdf-4e26-5c1b294cf81b_clueweb09B-desc.tar.bz2

#wget http://www.bigdatalab.ac.cn/benchmark/upload/download_source/af8c0582-9467-f031-6041-507bfa177ee2_clueweb09B-title.tar.bz2
#tar jxf af8c0582-9467-f031-6041-507bfa177ee2_clueweb09B-title.tar.bz2

#wget http://www.bigdatalab.ac.cn/benchmark/upload/download_source/717ebadf-eea1-aaf9-8754-787d7145d87f_wordembedding.tar.bz2
#tar jxf 717ebadf-eea1-aaf9-8754-787d7145d87f_wordembedding.tar.bz2

g++ main.cpp -o ./NN4IR NN4IR.cpp util.cpp Config.cpp -std=c++11  -O3 -w -funroll-loops -fopenmp 

./NN4IR -config config.ini > NN4IR.log &

#note that NN4IR.log will produce the test result, for rob04-title, the test result will be: MAP:0.2805, nDCG@20:0.4369, P@20: 0.3863.
tail -f -n 1000 NN4IR.log


## evaluation of the ranklist
#./trec_eval ./rob04-title/qrels.rob04.txt DRMM-LCH-IDF-rob04-title.ranklist -m all_trec

