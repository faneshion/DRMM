/*
*/

#include"util.h"

namespace nsnn4ir{

std::vector<double> linspace(double start_in,double end_in,int num_in){
    std::vector<double> linspaced(num_in);
    double delta = (end_in - start_in) / (num_in -1);
    for(int i = 0 ;i < num_in; ++i){
        linspaced[i] = start_in + delta * i;
    }
    linspaced[num_in-1] = end_in;
    return linspaced;
}

std::string double2str(double m,int precision){
    stringstream stream;
    if(precision<=0)    stream<<m;
    else    stream<<std::setprecision(precision)<<m;
    return stream.str();
}

void MSGPrint(char* fmt, ...){
    char time_info[64];
    time_t t = time(0);
    strftime(time_info, sizeof(time_info), "%X", localtime(&t));
    printf("[%s] ", time_info);
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
}

}
