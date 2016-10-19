/*
*/

#ifndef UTIL_H
#define UTIL_H

#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<vector>
#include<iterator>
#include<unordered_map>
#include<algorithm>
#include<iomanip>
//#include<limits>
#include<assert.h>
#include<cstdarg>
#include<cstdio>
using namespace std;

namespace nsnn4ir{
    typedef long long WINDEX;
    typedef long QINDEX;  //在Robust04的数据集的设置中，设置每个query的id为int

    struct _stWordInfo{
        string m_word;
        double m_df;
        double m_cf;
        double m_idf;
        double m_cfidf;
        _stWordInfo(string word="",double df=1.0,double cf=1.0,double idf=10.0,double cfidf=10.0):m_word(word),m_df(df),m_cf(cf),m_idf(idf),m_cfidf(cfidf){}
    };
    struct _stIRResult{
        double m_map;
        double m_patk;
        double m_ndcg;
        double m_erratk;
        //_stIRResult(_stIRResult & curr):m_map(curr.m_map),m_patk(curr.m_patk),m_ndcg(curr.m_ndcg){}
        _stIRResult(double cmap = 0,double cpatk = 0 ,double cndcg = 0,double cerratk=0):m_map(cmap),m_patk(cpatk),m_ndcg(cndcg),m_erratk(cerratk){}
        bool operator < (const _stIRResult & tmp){ return m_map < tmp.m_map ; }
        _stIRResult & operator = (const _stIRResult & tmp){ m_map = tmp.m_map; m_patk = tmp.m_patk; m_ndcg = tmp.m_ndcg; m_erratk = tmp.m_erratk; return *this;}
        _stIRResult operator + (const _stIRResult & tmp){ return _stIRResult(m_map+tmp.m_map,m_patk+tmp.m_patk,m_ndcg+tmp.m_ndcg,m_erratk+tmp.m_erratk); return *this;}
        _stIRResult & operator += (const _stIRResult & tmp){ m_map += tmp.m_map; m_patk += tmp.m_patk; m_ndcg += tmp.m_ndcg; m_erratk += tmp.m_erratk; return *this;}
        _stIRResult & operator /= (double f){ assert(f != 0); m_map /= f; m_patk /= f; m_ndcg /= f; m_erratk /= f; return *this;}
    };
    struct _stQRelInfo{
        int m_numofpos = 0;
        unordered_map<string,double> m_docrel;
        double m_idcg = 0.0;
    };
    struct _stDocInfo{
        int m_doclen;
        vector<WINDEX> m_docword;
        //unordered_map<WINDEX,double> m_docword;
    };
    std::vector<double> linspace(double start_in,double end_in,int num_in);

    std::string double2str(double m,int precision=0);

    void MSGPrint(char* fmt, ...);
}
#endif
