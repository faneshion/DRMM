# DRMM
This is an implementation of the paper ["A Deep Relevance Matching Model for Ad-hoc Retrieval"](http://www.bigdatalab.ac.cn/~gjf/papers/2016/CIKM2016a_guo.pdf). CIKM 2016.
![](./png/drmm.png)
# Guide to the code / usage instruction
this code is written in c++, and the dependence of Eigen is included. The abstract model structures are implemented in c++ class( see e.g. class NN4IR in file [NN4IR.cpp](./NN4IR.cpp) ). you can start training the robust04 title dataset by simply run 
<pre><code>sh build.sh</code></pre>
It will download the dataset and unpack it in the directory. All parameters passed by the [config](./config.ini) file which tells where the dataset lies. if you want to run other dataset or run other hyper-parameters, you can run
<pre><code>./NN4IR -config config_file</code></pre>
# Requirements
 * g++ 4.7 version or above( supporting c++11 )
