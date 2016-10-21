/*
 * date 0
 * author eshion
*/

#include "NN4IR.h"

using namespace nsnn4ir;

/*
   * the 5-fold splitation details conducted by Samuel.Huston and W.Bruce.Croft in their paper "Parameters Learned in the Comparison of Retrieval Models using Term Dependencies".
   * this 5-fold splitation made us fairly compare our model with their models.
*/
void NN4IR::BruceKFold(vector<QINDEX> vecQIndex,vector<vector<QINDEX>> &vecTrain){
    vecTrain.resize(5);
    switch(m_DataType){
        case DROB04:
            vecTrain[0] = {302,303,309,316,317,319,323,331,336,341,356,357,370,373,378,381,383,392,394,406,410,411,414,426,428,433,447,448,601,607,608,612,617,619,635,641,642,646,647,654,656,662,665,669,670,679,684,690,692,700};
            vecTrain[1] = {301,308,312,322,327,328,338,343,348,349,352,360,364,365,369,371,374,386,390,397,403,419,422,423,424,432,434,440,446,602,604,611,623,624,627,632,638,643,651,652,663,674,675,678,680,683,688,689,695,698};
            vecTrain[2] = {306,307,313,321,324,326,334,347,351,354,358,361,362,363,376,380,382,396,404,413,415,417,427,436,437,439,444,445,449,450,603,605,606,614,620,622,626,628,631,637,644,648,661,664,666,671,677,685,687,693};
            vecTrain[3] = {320,325,330,332,335,337,342,344,350,355,368,377,379,387,393,398,402,405,407,408,412,420,421,425,430,431,435,438,616,618,625,630,633,636,639,649,650,653,655,657,659,667,668,672,673,676,682,686,691,697};
            vecTrain[4] = {304,305,310,311,314,315,318,329,333,339,340,345,346,353,359,366,367,372,375,384,385,388,389,391,395,399,400,401,409,416,418,429,441,442,443,609,610,613,615,621,629,634,640,645,658,660,681,694,696,699}; 
            break;
        case DGOV2:
            vecTrain[0] = {712,722,731,740,749,750,751,759,764,765,774,775,782,784,785,786,788,789,794,798,801,805,823,824,825,832,833,835,845,850};
            vecTrain[1] = {704,708,709,715,718,738,743,755,756,761,762,763,772,779,792,802,806,808,809,814,816,818,821,822,831,836,839,840,843,847};
            vecTrain[2] = {702,703,705,706,707,710,720,721,724,725,728,730,733,734,736,737,747,757,770,771,773,797,803,804,810,812,820,828,834,844};
            vecTrain[3] = {701,713,714,717,723,726,727,729,732,735,760,769,778,780,787,790,791,793,795,799,811,813,819,826,827,837,842,846,848,849};
            vecTrain[4] = {711,716,719,739,741,742,744,745,746,748,752,753,754,758,766,767,768,776,777,781,783,796,800,807,815,817,829,830,838,841};
            break;
        case DCLU09:
            vecTrain[0] = {1,6,8,25,27,35,36,41,53,54,55,57,58,60,62,73,75,92,93,94,100,102,105,117,120,125,128,130,133,141};
            vecTrain[1] = {2,7,9,14,20,21,23,26,33,34,37,40,44,45,51,56,67,68,71,77,83,89,99,110,111,115,116,138,142,144};
            vecTrain[2] = {3,15,17,18,19,22,30,32,38,43,46,47,49,61,70,76,80,81,82,85,86,87,88,103,109,113,131,136,137,139};
            vecTrain[3] = {5,10,11,12,28,31,42,50,59,63,66,69,74,78,79,84,95,101,108,118,122,124,129,132,135,145,147,148,149,150};
            vecTrain[4] = {4,13,16,24,29,39,48,52,64,65,72,90,91,96,97,98,104,106,107,112,114,119,121,123,126,127,134,140,143,146};
            break;
        default:
			throw MyError(" Error data type, only Robust|Gov2|cluweb09B!");
    }
}
NN4IR::~NN4IR(){
}
void NN4IR::InitWordVec(const std::string & sfilename,bool binary){
    if(binary){
        long long row,col;
        FILE *f = fopen(sfilename.c_str(),"rb");
        if(!f){
			throw MyError(" Error: open word embedding file failed!");
        }
        fscanf(f,"%lld",&row);
        fscanf(f,"%lld",&col);
        unordered_map<string,VectorXd> mpOutVocab;

        for(int i = 0 ;i < row; ++i){
            std::string text = "";
            std::string lowtext="";
            while(1){
                char c = fgetc(f);
                if(feof(f) || c == ' ') break;
                text += c;
            }
            vector<float> currv(col,0.0);
            fread((char*)currv.data(),sizeof(float),col,f);
            VectorXd currvd = VectorXd::Zero(col);
            for(int j = 0 ; j < col; ++ j)  currvd(j) = currv[j];
            //currvd.normalize();
            if(m_words.find(text) == m_words.end()){
                mpOutVocab.insert(make_pair(text,currvd));
                continue;
            }
            m_W.insert(make_pair(m_words[text],currvd));
        }
        fclose(f);

        unordered_map<string,VectorXd>::iterator itF = mpOutVocab.begin();
        for(; itF != mpOutVocab.end(); ++itF){
            string stext =itF->first;
            std::transform(stext.begin(),stext.end(),stext.begin(),::tolower);
            if(m_words.find(stext) == m_words.end())    continue;
            if(m_W.find(m_words[stext]) == m_W.end()){
                m_W.insert(make_pair(m_words[stext],itF->second));
            }
        }
    }else{
        ifstream in(sfilename);
        string s,text;
        std::getline(in,s);
        size_t vocab_size,word_dim;
        istringstream iss(s);
        iss>>vocab_size>>word_dim;
        unordered_map<string,VectorXd> mpOutVocab;
        while(std::getline(in,s)){
            istringstream iss(s);
            iss >> text;
            VectorXd currv = VectorXd::Zero(word_dim);
            for(size_t i = 0 ;i < word_dim; ++i){
                iss >> currv[i];
            }
            if(m_words.find(text) == m_words.end()){
                mpOutVocab.insert(make_pair(text,currv));
                continue;
            }
            m_W.insert(make_pair(m_words[text],std::move(currv)));
        }
        in.close();
        in.clear();
        unordered_map<string,VectorXd>::iterator itF = mpOutVocab.begin();
        for(; itF != mpOutVocab.end(); ++itF){
            string stext =itF->first;
            std::transform(stext.begin(),stext.end(),stext.begin(),::tolower);
            if(m_words.find(stext) == m_words.end())    continue;
            if(m_W.find(m_words[stext]) == m_W.end()){
                m_W.insert(make_pair(m_words[stext],itF->second));
            }
        }
    }
}
void NN4IR::InitCorpInfo(const std::string & sDFCFfile,long long docNum){
    m_N = docNum;
    LoadTermDFCF(sDFCFfile);
}
void NN4IR::InitDocCorp(const std::string & sDocFile){
    assert(!m_vocab.empty() && !m_words.empty());
    ifstream fin(sDocFile.c_str());
    if(!fin){
		throw MyError(" Error: open document content file failed!");
    }
    string sline;
    int inum = 0;
    while(getline(fin,sline)){
        istringstream iss(sline); //query string
        vector<string> vecline;
        _stDocInfo currdocinfo;
        copy(istream_iterator<string>(iss),istream_iterator<string>(),back_inserter(vecline));
        if(vecline.empty()) continue;
        string did = vecline[0];
        int cdlen = atoi(vecline[1].c_str());
        currdocinfo.m_doclen = cdlen;
        currdocinfo.m_docword.reserve(cdlen);
        int countlen = 0;
        //if(m_DocCorp.find(did) != m_DocCorp.end()){cout<<"repeat doc "<<did<<" ...\n"; continue;}
        for(size_t j = 2 ; j < vecline.size(); ++j){
            //std::transform(vecline[j].begin(),vecline[j].end(),vecline[j].begin(),::tolower);
            size_t pos = vecline[j].find(":");
            string word = vecline[j].substr(0,pos);
            int count = atoi(vecline[j].substr(pos+1).c_str());
            //string word = vecline[j];
            if(m_words.find(word) == m_words.end())   continue;
            for(int i = 0 ; i < count; ++i)
            currdocinfo.m_docword.emplace_back(m_words[word]);
            //countlen += count;
        }
        m_DocCorp.insert(make_pair(did,currdocinfo));
        ++inum ;
    }
    MSGPrint("Init document corpus finished, doc count:%d ...\n", m_DocCorp.size());
    fflush(stdout);
}
void NN4IR::InitQueryCorp(const std::string & sQueryFile){
    assert(!m_words.empty() && !m_vocab.empty());
    ifstream fin(sQueryFile.c_str());
    if(!fin){
		throw MyError(" Error: open query content file failed!");
    }
    string sline;
    while(getline(fin,sline)){
        std::transform(sline.begin(),sline.end(),sline.begin(),::tolower);
        istringstream iss(sline); //query string
        vector<string> vecline;
        unordered_map<WINDEX,double> currqinfo;
        vector<WINDEX> currqinfo_seq;
        copy(istream_iterator<string>(iss),istream_iterator<string>(),back_inserter(vecline));
        if(vecline.empty()) continue;
        QINDEX cqindex = atol(vecline[0].c_str());
        for(size_t j = 1 ; j < vecline.size(); ++j){
            if(m_words.find(vecline[j]) == m_words.end()){   
                //printf(" query word : %s not found.\n", vecline[j].c_str()); 
                continue;}
            pair<unordered_map<WINDEX,double>::iterator,bool> ret = currqinfo.insert(make_pair(m_words[vecline[j]],1));
            if(!ret.second) ++ ret.first->second;

            currqinfo_seq.push_back(m_words[vecline[j]]);
        }
        m_QueryCorp.insert(make_pair(cqindex,currqinfo));
        m_seqQueryCorp.insert(make_pair(cqindex,currqinfo_seq));
    }
    MSGPrint("Init query corpus finished, query count:%d ...\n", m_QueryCorp.size());
    fflush(stdout);
}
void NN4IR::LoadDataSet(const std::string &sfilename,int topk,int pernegative,int num_of_instance){
    ifstream fin(sfilename.c_str());
    if(!fin){
		throw MyError(" Error: open ranklist file failed!");
    }
    string sline;
    unordered_set<string> errordoc ;//= {"clueweb09-enwp03-18-24469","clueweb09-en0010-99-41841","clueweb09-enwp03-10-24246"};
    int inum = 0;
    while(getline(fin,sline)){
        istringstream iss(sline); //query string
        vector<string> vecline;
        copy(istream_iterator<string>(iss),istream_iterator<string>(),back_inserter(vecline));

        //301 Q0 FR940425-2-00078 1 -6.66707536 galago
        if(vecline.size() != 6) continue;
        assert(vecline.size() == 6);
        QINDEX currqindex = atol(vecline[0].c_str()); // query no
        if(m_dataset.find(currqindex) == m_dataset.end()){
            m_dataset.insert(make_pair(currqindex,vector<string>()));
        }
        if(errordoc.find(vecline[2]) != errordoc.end()) continue;
        if((int)m_dataset[currqindex].size() < topk){ ++inum;     m_dataset[currqindex].push_back(vecline[2]);  }
    }
    MSGPrint("Init data set finished, records count:%d ...\n", inum);
    fflush(stdout);

    // generate pair-wise instance: in order to not be dominated by the queries with more relevant document, we made a trick that the query with more relevant document will generate less pairs.
    vector<int> all_pos;
    for(auto iter = m_relinfo.begin(); iter != m_relinfo.end(); ++ iter)    all_pos.push_back(iter->second.m_numofpos);
    std::sort(all_pos.begin(),all_pos.end(),[](const int a,const int b) -> bool{ return a < b; });
    int avg_pos_34 = all_pos[all_pos.size() * 3 / 4];
    int avg_pos_24 = all_pos[all_pos.size() / 2];
    int avg_pos_14 = all_pos[all_pos.size() / 4];
    //for(auto mmm = all_pos.begin(); mmm != all_pos.end(); ++ mmm)   cout<<*mmm<<" ";
    //cout<<"\n avg_pos_14:"<<avg_pos_14<<" avg_pos_24:"<<avg_pos_24<<" avg_pos_34:"<<avg_pos_34<<endl;
    random_device rdp,rdn;
    mt19937 ren(rdn());
    long totalinstance = 0;
    for(auto iter = m_dataset.begin(); iter != m_dataset.end(); ++ iter){
        QINDEX cqindex = iter->first;
        m_QInstance.insert(make_pair(cqindex,vector<pair<string,string>>()));
        map<double,vector<string>,std::greater<double>> currsamples;
        int num_pos_currquery = 0;
        for(auto viter = iter->second.begin(); viter != iter->second.end(); ++ viter){
            double currlabel = 0.0;
            if(m_relinfo[cqindex].m_docrel.find(*viter) != m_relinfo[cqindex].m_docrel.end()){
                currlabel = m_relinfo[cqindex].m_docrel[*viter];
            }
            if(currsamples.find(currlabel) == currsamples.end())    currsamples.insert(make_pair(currlabel,vector<string>()));
            if(currlabel > 0)  ++ num_pos_currquery;
            currsamples[currlabel].push_back(*viter);
        }
        int curr_pernegative = pernegative;
        int curr_num_of_instance = num_of_instance;
        if(num_pos_currquery <= avg_pos_14)    {   curr_pernegative *= 5; curr_num_of_instance *= 5; }
        else if(num_pos_currquery <= avg_pos_24)    {   curr_pernegative *= 3; curr_num_of_instance *= 3; }
        else if(num_pos_currquery <= avg_pos_34)    {   curr_pernegative *= 1.5; curr_num_of_instance *= 1.5; }
        int total_instance = 0;
        for(auto csiter = currsamples.begin(); csiter != currsamples.end(); ++ csiter){
            for(auto next_csiter = std::next(csiter,1); next_csiter != currsamples.end(); ++ next_csiter)
                total_instance += csiter->second.size() * next_csiter->second.size();
        }
        total_instance = total_instance < curr_num_of_instance ? total_instance : curr_num_of_instance;
        unordered_set<string> allready;
        int curr_q_instance = 0;
        for(auto csiter = currsamples.begin(); csiter != currsamples.end(); ++ csiter){
            for(int i = 0 ; i < csiter->second.size(); ++ i){
                vector<string> negatives;
                for(auto next_csiter = std::next(csiter,1); next_csiter != currsamples.end(); ++ next_csiter)
                    for(int j = 0 ; j < next_csiter->second.size(); ++ j){
                        negatives.push_back(next_csiter->second[j]);
                    }
                uniform_int_distribution<int> uin(0,negatives.size()-1);
                for(int j = 0 ; j < negatives.size() && j < curr_pernegative; ){
                    if(allready.size() >= total_instance) break;
                    int inegdoc = uin(ren);
                    //int inegdoc = j;
                    string skey = csiter->second[i]+negatives[inegdoc];
                    if(allready.find(skey) != allready.end()) continue;
                    allready.insert(skey);
                    m_QInstance[cqindex].push_back(make_pair(csiter->second[i],negatives[inegdoc]));
                    ++ totalinstance;
                    ++ curr_q_instance;
                    ++ j;
                }
                if(allready.size() >= total_instance) break;
            }
            if(allready.size() >= total_instance) break;
        }
    }
    MSGPrint("Generate pair-wise samples: %d .\n", totalinstance);

    fflush(stdout);
}
void NN4IR::InitGroundTruth(const std::string &sRelFile,const std::string &sIDCGFile,double relevance_level){
    string sline;
    ifstream fin0(sIDCGFile.c_str(),ios::in);
    if(!fin0){
		throw MyError(" Error: open idcg file failed!");
    }
    while(getline(fin0,sline)){
        istringstream iss(sline); //query string
        _stQRelInfo tmpqrelinfo;
        vector<string> vecline;
        copy(istream_iterator<string>(iss),istream_iterator<string>(),back_inserter(vecline));
        assert(vecline.size() == 2);
        QINDEX currqindex = atol(vecline[0].c_str()); // query no
        double idcg = atof(vecline[1].c_str());
        tmpqrelinfo.m_idcg = idcg;
        assert(m_relinfo.find(currqindex) == m_relinfo.end());
        m_relinfo.insert(make_pair(currqindex,tmpqrelinfo));
    }
    fin0.close();
    ifstream fin(sRelFile.c_str());
    if(!fin){
		throw MyError(" Error: open relevant file failed!");
    }
    while(getline(fin,sline)){
        istringstream iss(sline); //query string
        vector<string> vecline;
        copy(istream_iterator<string>(iss),istream_iterator<string>(),back_inserter(vecline));
        assert(vecline.size() == 4);
        QINDEX currqindex = atol(vecline[0].c_str()); // query no
        double clabel = atof(vecline[3].c_str());
        clabel = clabel >= relevance_level ? clabel : 0;
        m_MAX_JUDGEMENT = clabel > m_MAX_JUDGEMENT ? clabel : m_MAX_JUDGEMENT;
        assert(m_relinfo.find(currqindex) != m_relinfo.end());
        m_relinfo[currqindex].m_docrel.insert(make_pair(vecline[2],clabel));
        if(clabel >= relevance_level)   m_relinfo[currqindex].m_numofpos += 1;
    }
    fin.close();
    m_MAX_JUDGEMENT = 4;
    MSGPrint("Init groundtruth data finished, records num:%u.\n",m_relinfo.size());

    fflush(stdout);
}
void NN4IR::LoadTermDFCF(const std::string &sfilename){
    fstream fin(sfilename,std::ios::in);
    if(!fin){
		throw MyError(" Error: open term df&cf file failed!");
    }
    string sline;
    m_VocabSize = 1;
    while(getline(fin,sline)){
        istringstream iss(sline);
        string word;
        double currdf,currcf;
        iss >> word >> currdf >> currcf;
        if(currdf <= 0) currdf = 1;
        if(currcf <= 0) currcf = 1;
        assert(currdf > 0 && currcf > 0);
        std::transform(word.begin(),word.end(),word.begin(),::tolower);

        assert(!word.empty());
        //cout<<word<<":"<<count<<endl;
        if(m_words.find(word) == m_words.end()){
            m_words[word] = m_VocabSize;
            double curridf = log((double)(m_N + 1) / currdf);
            assert(curridf > 0);
            double currcfidf = currcf * curridf;
            m_vocab.insert(make_pair(m_VocabSize,_stWordInfo(word,currdf,currcf,curridf,currcfidf)));
            ++m_VocabSize;
        }
    }
    MSGPrint("Init Term DF & CF finished, vocab size:%d.\n", m_words.size());
    fflush(stdout);
}
void NN4IR::kFold(vector<QINDEX> vecQIndex,vector<vector<QINDEX>> & vecTrain,int nFold,bool shuffle){
    vecTrain.clear();
    vecTrain.resize(nFold);
    if(shuffle){
        std::random_shuffle(vecQIndex.begin(),vecQIndex.end());
    }
    vector<QINDEX>::iterator iter = vecQIndex.begin();
    while(iter != vecQIndex.end()){
        for(int i = 0 ; i < nFold; ++i){
            vecTrain[i].push_back(*iter);
            ++iter;
            if(iter == vecQIndex.end())
                return;
        }
    }
}

void NN4IR::InitTopKNeiB(){
    m_QTopKNeighbor.clear();
    unordered_set<WINDEX> allquerywords;
    for(auto iter = m_QueryCorp.begin(); iter != m_QueryCorp.end(); ++ iter){
        for_each(iter->second.begin(),iter->second.end(),[&](const pair<WINDEX,double> & k){    
                allquerywords.insert(k.first);
                });
    }
    for(unordered_set<WINDEX>::iterator itQW = allquerywords.begin(); itQW != allquerywords.end(); ++ itQW){
        m_QTopKNeighbor.insert(make_pair(*itQW,unordered_map<WINDEX,double>()));
        if(m_W.find(*itQW) == m_W.end()){
            m_QTopKNeighbor[*itQW].insert(make_pair(*itQW,1.0));
            continue;
        }
        VectorXd vecQW = m_W[*itQW]; 
        unordered_map<WINDEX,VectorXd>::iterator it = m_W.begin();
        double curr_max = 0;
        for(; it != m_W.end(); ++ it){
            double simi = vecQW.dot(it->second);
            simi /= vecQW.norm() * it->second.norm();
            if(simi > curr_max) curr_max = simi;
            m_QTopKNeighbor[*itQW].insert(make_pair(it->first,simi));
        }
        //assert(curr_max == 1);
        if(curr_max != 0){
            for(auto itCWN = m_QTopKNeighbor[*itQW].begin(); itCWN != m_QTopKNeighbor[*itQW].end(); ++ itCWN){
                itCWN->second = itCWN->second / curr_max;
            }
        }
    }
    MSGPrint("Init query topk neighbors finished.\n");
    fflush(stdout);
}

bool NN4IR::Simi_evaluate(const double relevance_level,const QINDEX & qindex,const unordered_map<string,double> & scores,int evaluatenum ,int topk,_stIRResult &evalres){
    if(m_relinfo[qindex].m_numofpos <= 0)   return false;
    assert(topk > 0 && evaluatenum > 0);
    if(m_relinfo[qindex].m_numofpos <= 0){  
        //cout<<"query "<<qindex<<" without positive docs ..."<<endl; fflush(stdout);   
        return false; 
    }
    evaluatenum = (int)scores.size() < evaluatenum ? (int)scores.size() : evaluatenum;
    topk  = (int)scores.size() < topk ? (int)scores.size() : topk;
    vector<pair<double,double>> resinfo(scores.size(),pair<double,double>(0,0)); //pair: label , score
    int inum = 0;
    for(auto iter = scores.begin() ; iter != scores.end(); ++iter){
        double clabel = 0;
        if(m_relinfo[qindex].m_docrel.find(iter->first) != m_relinfo[qindex].m_docrel.end())  clabel = m_relinfo[qindex].m_docrel[iter->first];
        //clabel = clabel >= relevance_level ? clabel : 0;
        resinfo[inum].first = clabel;
        resinfo[inum].second = iter->second;
        ++inum;
    }
    std::sort(resinfo.begin(),resinfo.end(),[](const pair<double,double> & a,const pair<double,double> &b) -> bool{return a.second > b.second;});   // 相似度，jiang序
    //map
    double rel_so_far = 0;
    for(int i = 0 ; i < evaluatenum; ++i){
        if(resinfo[i].first >= relevance_level){
             ++ rel_so_far ;
            evalres.m_map += (double) rel_so_far / (double) ( i + 1 );
        }
    }
    evalres.m_map /= m_relinfo[qindex].m_numofpos;
    //p@k
    for(int j = 0 ; j < topk; ++j){
        if(resinfo[j].first >= relevance_level)
            ++evalres.m_patk;
    }
    evalres.m_patk /= topk;
    //err@k
    double fdecay = 1.0;
    double max_judge = pow(2,m_MAX_JUDGEMENT);
    for(int j = 0 ; j < topk; ++j){
        double r = (pow(2,resinfo[j].first) - 1) / max_judge;
        evalres.m_erratk += r*fdecay /(j + 1);
        fdecay *= (1 - r);
    }
    //ndcg
    double dcg_score = 0.0,idcg_score = 0.0;
    for(int j = 0 ; j < topk; ++j){
        if(resinfo[j].first >= relevance_level)
            dcg_score += ( pow(2,resinfo[j].first) - 1 ) / (double)log((double)(j + 2));
    }
    /*
    std::stable_sort(resinfo.begin(),resinfo.end(),[](const pair<double,double> & a,const pair<double,double> &b) -> bool{return a.first > b.first;});   // 距离 ，降序
    for( int j = 0 ; j < topk; ++j){
        if(resinfo[j].first >= relevance_level)
            idcg_score += (pow(2,resinfo[j].first)-1) / (double)log((double)(j + 2));
    }
    if(idcg_score != 0)
        evalres.m_ndcg = dcg_score / idcg_score;
    */
    if(m_relinfo[qindex].m_idcg != 0)
        evalres.m_ndcg = dcg_score / m_relinfo[qindex].m_idcg;
    return true;
}
void NN4IR::EvaluateFile(const std::string & sfilename){
    ifstream fin(sfilename.c_str());
    if(!fin){
		throw MyError(" Error: open evaluation file failed!");
    }
    string sline;
    unordered_map<QINDEX,vector<pair<string,double>>> currRes;
    while(getline(fin,sline)){
        istringstream iss(sline); //query string
        vector<string> vecline;
        unordered_map<string,double> cdocrel;
        copy(istream_iterator<string>(iss),istream_iterator<string>(),back_inserter(vecline));

        //if(vecline.empty()) continue;
        //301 Q0 FR940425-2-00078 1 -6.66707536 galago
        assert(vecline.size() == 6);
        QINDEX currqindex = atol(vecline[0].c_str()); // query no
        double cscore = atof(vecline[4].c_str());

        if(currRes.find(currqindex) == currRes.end()){
            currRes.insert(make_pair(currqindex,vector<pair<string,double>>()));
        }
        currRes[currqindex].push_back(make_pair(vecline[2],cscore));
    }

    _stIRResult eval;
    int valid_topic_num = 0;
    for(auto iter = currRes.begin(); iter != currRes.end(); ++ iter){
        unordered_map<string,double> currscore;
        for(size_t i = 0 ; i < iter->second.size(); ++ i)
            currscore.insert(make_pair(iter->second[i].first,iter->second[i].second));
        _stIRResult tmpeval;
        if(Simi_evaluate(1,iter->first,currscore,1000,20,std::ref(tmpeval))){
            eval += tmpeval;
            ++ valid_topic_num;
        //}else{
            //printf("qindex:%d evaluate Error\n",iter->first);
        }
    }
    eval /= valid_topic_num;
    MSGPrint("Evaluating Result: valid-topic-num:%d, map:%.4f, P@k:%.4f, nDCG@k:%.4f, ERR@k:%.4f\n",valid_topic_num,eval.m_map,eval.m_patk,eval.m_ndcg,eval.m_erratk);
}
void NN4IR::RunningMultiThread(int nFold,int maxiter){
    if(m_dataset.empty()){
		throw MyError(" Error: no dataset found!");
        return;
    }
    long a=0,b=0,c=0;
    vector<QINDEX> vecQIndex(m_dataset.size());
    for(auto iter = m_dataset.begin(); iter != m_dataset.end(); ++ iter) vecQIndex[a++] = iter->first; 
    vector<vector<QINDEX>> vecFolds;
    BruceKFold(vecQIndex,vecFolds);
    //const int nVecDim = m_W.begin()->second.size();
    const int nVecDim = 2; //term gating paramerter size

    const int vW1Size = 2; // layer size
    vector<int> vW1SizeInfo = {30,5,1}; // layer parameters W1: 30*5, W2: 5*1 

    vector<RMatrixXd> vW1_grad(vW1Size);
    vector<VectorXd>    vW3_grad(vW1Size);
    for(a = 0 ; a < vW1Size; ++ a){
        vW1_grad[a] = RMatrixXd::Zero(vW1SizeInfo[a],vW1SizeInfo[a+1]);
        vW3_grad[a].resize(vW1SizeInfo[a+1]);
        vW3_grad[a].setZero();
    }
    VectorXd vW2_grad = VectorXd::Zero(nVecDim);

    _stIRResult result;
    vector<double> resFoldTest(nFold);

    for(a = 0 ; a < nFold; ++a){ // 迭代5轮，每次留第a个fold作为测试
        double lr_w1 = m_lr_w1;
        double lr_w2 = m_lr_w2;
        vector<RMatrixXd> m_vW1(vW1Size);
        vector<VectorXd> m_vW3(vW1Size);
        for(b = 0 ; b < vW1Size; ++ b)  {
            m_vW1[b] = RMatrixXd::Random(vW1_grad[b].rows(),vW1_grad[b].cols());
            m_vW1[b] /= 10.0;
            m_vW3[b] = VectorXd::Random(vW3_grad[b].size());
        }
        VectorXd m_vW2 = VectorXd::Random(nVecDim);
        m_vW2 /= 100.0;
        double best_map = 0.0;
        vector<RMatrixXd> m_vW1_best = m_vW1;
        vector<VectorXd> m_vW3_best = m_vW3;
        VectorXd m_vW2_best = m_vW2;
        vector<pair<QINDEX,pair<string,string>>> allTrainInstance;
        for(b = 0 ; b < nFold; ++b){
            if(b == a)  continue;
            for(c = 0 ; c < vecFolds[b].size(); ++c){
                QINDEX qIndex = vecFolds[b][c];
                if(m_relinfo[qIndex].m_numofpos <= 0)   continue;
                if( m_QInstance.find(qIndex) == m_QInstance.end())  continue;
                for(auto iter = m_QInstance[qIndex].begin(); iter != m_QInstance[qIndex].end(); ++ iter){
                    allTrainInstance.push_back(make_pair(qIndex,make_pair(iter->first,iter->second)));
                }
            }
        }
        std::random_shuffle(allTrainInstance.begin(),allTrainInstance.end());
        MSGPrint(" start to train %d instance. \n", allTrainInstance.size());
        fflush(stdout);
        long total_case_number = allTrainInstance.size();
        long  max_iterloop= maxiter * total_case_number / m_mini_batch;
        long iterloop = 0;
        bool bfinished = false;
        while(!bfinished){
            double looploss = 0.0;
            int batch_size = total_case_number / m_mini_batch ; //注意，这里丢弃了部分的case，使得下面的omp 可以编译通过
            for(b = 0 ; b < batch_size; ++ b){
                for(c = 0 ; c < vW1Size; ++ c){
                    vW1_grad[c].setZero();
                    vW3_grad[c].setZero();
                }
                vW2_grad.setZero();
                double batchloss = 0.0;
                #pragma omp parallel for
                for(c = 0 ; c < m_mini_batch; ++c){ //每 m_mini_batch 个instance  更新参数
                    int curr_instance_index = b * m_mini_batch + c;
                    pair<QINDEX,pair<string,string>> & curr_instance = allTrainInstance[curr_instance_index];
                    vector<RMatrixXd>    vW1_pgd(vW1Size),vW1_ngd(vW1Size);
                    vector<VectorXd>     vW3_pgd(vW1Size),vW3_ngd(vW1Size);
                    for(long d = 0 ; d < vW1Size; ++ d){
                        vW1_pgd[d] = RMatrixXd::Zero(vW1_grad[d].rows(),vW1_grad[d].cols());
                        vW1_ngd[d] = RMatrixXd::Zero(vW1_grad[d].rows(),vW1_grad[d].cols());
                        vW3_pgd[d].resize(vW3_grad[d].size());
                        vW3_ngd[d].resize(vW3_grad[d].size());
                        vW3_pgd[d].setZero();
                        vW3_ngd[d].setZero();
                    }
                    VectorXd vW2_pgd = VectorXd::Zero(nVecDim),vW2_ngd = VectorXd::Zero(nVecDim); 
                    double score1 = 0, score2 = 0;
                    score1 = NNScore_LCH_IDF(curr_instance.first,curr_instance.second.first,m_vW1,m_vW2,m_vW3,vW1_pgd,vW2_pgd,vW3_pgd,true); //positive score
                    score2 = NNScore_LCH_IDF(curr_instance.first,curr_instance.second.second,m_vW1,m_vW2,m_vW3,vW1_ngd,vW2_ngd,vW3_ngd,true); //negative score
                    double currloss = 0.1 - score1 + score2;
                    if(currloss > 0){ //calculate gradient , update parameters
                        for(long d = 0 ; d < vW1Size; ++ d){
                            vW1_grad[d] += lr_w1 * (vW1_pgd[d] - vW1_ngd[d]);
                            vW3_grad[d] += lr_w1 * (vW3_pgd[d] - vW3_ngd[d]);
                        }
                        vW2_grad += lr_w2 * (vW2_pgd - vW2_ngd);
                        looploss += currloss;
                        batchloss += currloss;
                    }
                }
                for(c = 0 ; c < vW1Size; ++ c){
                    m_vW1[c] += vW1_grad[c] / m_mini_batch;
                    m_vW3[c] += vW3_grad[c] / m_mini_batch;
                }
                m_vW2 += vW2_grad / m_mini_batch;
                ++iterloop;
                // learining rate decay
                if( iterloop > 0 && (iterloop % m_lr_decay_interval == 0)){
                    lr_w1 = m_lr_w1 * (1.0 - iterloop / (1.0 + max_iterloop));
                    if(lr_w1 < m_lr_w1 * 0.0001) lr_w1 = m_lr_w1 * 0.001;
                    lr_w2 = m_lr_w2 * (1.0 - iterloop / (1.0 + max_iterloop));
                    if(lr_w2 < m_lr_w2 * 0.0001) lr_w2 = m_lr_w2 * 0.001;
                }
                // show train and test performance 
                if(iterloop > 0 && iterloop % m_show_interval == 0){
                    MSGPrint("iterator:%dk|%dk,lr(w1):%.6f, Loss:%.6f\n", int(iterloop/1000), int(max_iterloop/1000), lr_w1, batchloss/m_mini_batch );
                    //training performance
                    /*
                    int iValidTrainQuery = 0;
                    _stIRResult currTrainRes;
                    for(b = 0 ; b < nFold; ++b){
                        if(b == a)  continue;
                        #pragma omp parallel for
                        for(c = 0 ; c < vecFolds[b].size(); ++c){
                            QINDEX qIndex = vecFolds[b][c];
                            unordered_map<string,double> currscore;
                            assert(m_dataset.find(qIndex) != m_dataset.end());
                            for(auto iter = m_dataset[qIndex].begin(); iter != m_dataset[qIndex].end(); ++ iter){
                                vector<RMatrixXd> vW1_pgd(vW1Size);
                                vector<VectorXd> vW3_pgd(vW1Size);
                                for(long d = 0 ; d < vW1Size; ++ d){
                                    vW1_pgd[d] = RMatrixXd::Zero(vW1_grad[d].rows(),vW1_grad[d].cols());
                                    vW3_pgd[d].resize(vW3_grad[d].size());
                                    vW3_pgd[d].setZero();
                                }
                                VectorXd vW2_pgd = VectorXd::Zero(nVecDim);
                                double score = NNScore_LCH_IDF(qIndex,*iter,m_vW1,m_vW2,m_vW3,vW1_pgd,vW2_pgd,vW3_pgd,false);
                                currscore.insert(make_pair(*iter,score));
                            }
                            _stIRResult tmpeval;
                            if(Simi_evaluate(1,qIndex,currscore,1000,20,std::ref(tmpeval))){
                                currTrainRes += tmpeval;
                                ++iValidTrainQuery;
                            }
                        }
                    }
                    currTrainRes /= iValidTrainQuery;
                    MSGPrint("Train MAP:%.4f\tP@k:%.4f\tnDCG@k:%.4f\n",currTrainRes.m_map,currTrainRes.m_patk,currTrainRes.m_ndcg);
                    */
                    int iValidTestQuery = 0;
                    _stIRResult currTestRes;
                    #pragma omp parallel for
                    for(b = 0 ; b < vecFolds[a].size(); ++ b){
                        QINDEX qIndex = vecFolds[a][b];
                        unordered_map<string,double> currscore;
                        assert(m_dataset.find(qIndex) != m_dataset.end());
                        for(auto iter = m_dataset[qIndex].begin(); iter != m_dataset[qIndex].end(); ++ iter){
                            vector<RMatrixXd> vW1_pgd(vW1Size);
                            vector<VectorXd> vW3_pgd(vW1Size);
                            for(long d = 0 ; d < vW1Size; ++ d){
                                vW1_pgd[d] = RMatrixXd::Zero(vW1_grad[d].rows(),vW1_grad[d].cols());
                                vW3_pgd[d].resize(vW3_grad[d].size());
                                vW3_pgd[d].setZero();
                            }
                            VectorXd vW2_pgd = VectorXd::Zero(nVecDim);
                            double score = NNScore_LCH_IDF(qIndex,*iter,m_vW1,m_vW2,m_vW3,vW1_pgd,vW2_pgd,vW3_pgd,false);
                            currscore.insert(make_pair(*iter,score));
                        }
                        _stIRResult tmpeval;
                        if(Simi_evaluate(1,qIndex,currscore,1000,20,std::ref(tmpeval))){
                            currTestRes += tmpeval;
                            ++iValidTestQuery;
                        }
                    }
                    currTestRes /= iValidTestQuery;
                    MSGPrint("\tFold: %d, Test MAP:%.4f, P@k:%.4f, nDCG@k:%.4f\n", a, currTestRes.m_map, currTestRes.m_patk, currTestRes.m_ndcg);
                    fflush(stdout);
                    if(currTestRes.m_map > best_map){
                        best_map = currTestRes.m_map;
                        m_vW1_best = m_vW1;
                        m_vW2_best = m_vW2;
                        m_vW3_best = m_vW3;
                    }
                }
                if(iterloop >= max_iterloop){
                    bfinished = true;
                    break;
                }
            }
            if(looploss <= 0)   break;
        }
        _stIRResult cTestRes;
        int iValidTestQuery = 0;
        #pragma omp parallel for
        for(b = 0 ; b < vecFolds[a].size(); ++ b){
            QINDEX qIndex = vecFolds[a][b];
            omp_set_lock(&lock);
            if(m_RankInfo.find(qIndex) == m_RankInfo.end()) m_RankInfo.insert(make_pair(qIndex,multimap<double,string,std::greater<double>>()));
            omp_unset_lock(&lock);
            unordered_map<string,double> currscore;
            assert(m_dataset.find(qIndex) != m_dataset.end());
            for(auto iter = m_dataset[qIndex].begin(); iter != m_dataset[qIndex].end(); ++ iter){
                vector<RMatrixXd> vW1_pgd(vW1Size);
                vector<VectorXd> vW3_pgd(vW1Size);
                for(long d = 0 ; d < vW1Size; ++ d){  
                    vW1_pgd[d] = RMatrixXd::Zero(vW1_grad[d].rows(),vW1_grad[d].cols());
                    vW3_pgd[d].resize(vW3_grad[d].size());
                    vW3_pgd[d].setZero();
                }
                VectorXd vW2_pgd = VectorXd::Zero(nVecDim);
                double score = NNScore_LCH_IDF(qIndex,*iter,m_vW1_best,m_vW2_best,m_vW3_best,vW1_pgd,vW2_pgd,vW3_pgd,false);
                omp_set_lock(&lock);
                m_RankInfo[qIndex].insert(make_pair(score,*iter));
                omp_unset_lock(&lock);
                currscore.insert(make_pair(*iter,score));
                assert(m_dataset.find(qIndex) != m_dataset.end());
            }
            _stIRResult tmpeval;
            if(Simi_evaluate(1,qIndex,currscore,1000,20,std::ref(tmpeval))){
                cTestRes += tmpeval;
                ++iValidTestQuery;
            }
        }
        cTestRes /= iValidTestQuery;

        MSGPrint("Fold %d test result: map:%.4f, p@k:%.4f, nDCG@k:%.4f, ERR@k:%.4f, valid-topic-num:%d\n",
                a, cTestRes.m_map, cTestRes.m_patk, cTestRes.m_erratk, iValidTestQuery);
        fflush(stdout);

        result += cTestRes;
        
        //record resul info just for easy fitting excel
        resFoldTest[a] = cTestRes.m_map;
    }
    result /= nFold;
    MSGPrint("Final average result: MAP:%.4f, P@20:%.4f, nDCG@20:%.4f, ERR@20:%.4f\n\tFold MAPs:", result.m_map, result.m_patk, result.m_ndcg, result.m_erratk);
    for_each(resFoldTest.begin(),resFoldTest.end(),[](const double & k){cout<<k<<"\t";});
    cout<<endl;
    fflush(stdout);
    return ;
}

double NN4IR::NNScore_LCH_IDF(const QINDEX & qindex,const string & currdoc,const vector<RMatrixXd> & vW1,const VectorXd & vW2,const vector<VectorXd> & vW3,vector<RMatrixXd> & vW1_gd,VectorXd & vW2_gd,vector<VectorXd> & vW3_gd,bool bTrain){
    assert(vW1.size() == vW1_gd.size() && vW2.size() == vW2_gd.size() && vW3.size() == vW3_gd.size() && vW1.size() == vW3.size());
    long a,b,c,d,e;
    double eps = 1e-4;
    double score = 0;
    const int nQSize = m_seqQueryCorp[qindex].size();
    const int nVecDim = vW2.size();
    const int vW1Size = vW1.size();
    assert(vW1Size > 0 );
    const vector<WINDEX> & currsequence = m_DocCorp[currdoc].m_docword;
    const int iDocLen = currsequence.size();
    if(iDocLen == 0 || nQSize == 0)    return std::numeric_limits<double>::lowest();
    RMatrixXd mm = RMatrixXd::Constant(nQSize,vW1[0].rows(),0);
    const int iHalfSize = vW1[0].rows();
    for(a = 0 ; a < nQSize; ++ a){
        const WINDEX & cqword = m_seqQueryCorp[qindex][a];
        for(b = 0 ; b < iDocLen; ++ b){
            double simi1 = 0.0;
            if(cqword == currsequence[b]){
                simi1 = 1.0;
            }else if(m_QTopKNeighbor.find(cqword) != m_QTopKNeighbor.end() && m_QTopKNeighbor[cqword].find(currsequence[b]) != m_QTopKNeighbor[cqword].end()){
                simi1 = m_QTopKNeighbor[cqword][currsequence[b]];
            }
            int iw1 = (simi1 + 1.0 ) / 2.0 * (iHalfSize - 1);
            assert( iw1 >= 0 && iw1 < iHalfSize);
            mm(a,iw1) += 1.0;
        }
    }
    double weight_sum = 0.0;
    VectorXd vQWeight = VectorXd::Zero(nQSize); //zi
    VectorXd vW2_gd_tmp = VectorXd::Zero(nVecDim);
    for( a = 0 ; a < nQSize; ++ a){
        WINDEX cqword = m_seqQueryCorp[qindex][a];
        double cqw = 2.0;
        cqw = exp(vW2(0) * (m_vocab[cqword].m_idf)); 
        if(bTrain){
            vW2_gd_tmp(0) += cqw * (m_vocab[cqword].m_idf);
        }
        vQWeight(a) = cqw; 
        weight_sum += vQWeight(a);
    }

    assert(weight_sum != 0);
    vQWeight /= weight_sum;
    vW2_gd_tmp /= weight_sum;

    vector<VectorXd> vHi(vW1Size + 1);
    vector<VectorXd> sigma(vW1Size);
    vHi[0].resize(vW1[0].rows());
    for(a = 1 ; a < vW1Size + 1; ++ a){
        vHi[a].resize(vW1[a-1].cols()); 
        sigma[a-1].resize(vHi[a].size());
    }
    for(a = 0; a < mm.rows(); ++ a){
        const WINDEX & cqword = m_seqQueryCorp[qindex][a];
        vHi[0].setZero();
        for(int b = 0 ; b < vHi[0].size(); ++ b){
            vHi[0](b) = log10(mm(a,b) + 1);
        }
        //feed forward
        for(b = 1 ; b < vW1Size + 1; ++ b){ 
            vHi[b].setZero();
            vHi[b] = vHi[b-1] * vW1[b-1] + vW3[b-1];
            vHi[b] = m_actfunc.forward(vHi[b]);
        }
        score += vHi[vW1Size](0) * vQWeight(a);
        //backpropagation
        if(bTrain){
            for(b = vW1Size; b > 0; --b){
                sigma[b-1].setZero();
                if(b == vW1Size){    
                    VectorXd f_gd = m_actfunc.backward(vHi[b]);
                    sigma[b-1] = vQWeight(a) * f_gd;
                }else{
                    VectorXd f_gd = m_actfunc.backward(vHi[b]);
                    sigma[b-1] = (sigma[b] * vW1[b].transpose()).cwiseProduct(f_gd);
                }
                vW1_gd[b-1] += vHi[b-1].transpose() * sigma[b-1];
                vW3_gd[b-1] += sigma[b-1];
            }
            vW2_gd(0) += vHi[vW1Size](0) * vQWeight(a) * ((m_vocab[cqword].m_idf) - vW2_gd_tmp(0));
        }
    }
    return score;
}

void NN4IR::GetRanklist(const char* filename){
    ofstream fout(filename, ios::out);
    for(auto itQ = m_RankInfo.begin(); itQ != m_RankInfo.end(); ++ itQ){
        int irank = 0;
        for(auto itRanklist = itQ->second.begin(); itRanklist != itQ->second.end(); ++ itRanklist){
            fout<<itQ->first<<"\tQ0\t"<<itRanklist->second<<"\t"<<irank<<"\t"<<itRanklist->first<<"\tNN4IR-LCH-IDF\n";
            ++irank;
        }
    }
    fout.close();
    fout.clear();
}
