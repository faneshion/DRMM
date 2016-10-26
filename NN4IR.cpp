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
        case LETORMQ2007:
            vecTrain[0] = {8427,8424,9669,8422,9200,8158,9472,8157,9364,8627,8383,8382,8623,9466,8987,9670,9169,8545,9999,8013,7968,8785,8308,8096,8095,9589,9289,8092,9781,9021,9538,8869,8710,9688,9939,8713,9564,8908,8835,8386,9568,9680,8653,8652,9685,9684,9958,8149,8430,9308,9471,9067,8436,8141,9373,8632,8630,9708,9487,9881,8334,9989,9722,8023,8020,9174,9423,8024,8537,9906,9904,9367,8463,8879,8978,8979,8701,9570,9116,9282,8709,9578,9009,9923,8233,9768,9260,9098,9878,8602,8605,9498,8445,9016,8528,7979,8687,9012,9991,8324,8327,8137,9147,9302,8524,9384,9831,8889,8935,8541,8886,8881,9876,9464,9454,8968,9786,9865,9547,9772,8732,9445,9540,9726,8041,9725,9313,8045,8688,8754,9086,8750,8864,8519,9997,8456,9489,8283,9001,8281,8126,8352,8353,8354,9312,8288,8120,9502,9394,8897,8894,8893,9571,9559,8816,8812,9963,9550,9263,9798,9968,8726,8053,8218,9732,9845,8749,8748,9241,9243,9765,9639,9928,9992,9747,8290,9329,8297,8005,8299,8342,8469,9326,8125,9314,8196,8195,8246,9611,9640,9089,9402,8617,8616,9973,8610,9507,9404,9835,8592,8593,9979,9335,9130,8372,9566,8068,9070,9450,8064,9138,8109,8063,9855,8379,9681,9629,8577,8268,9932,8475,9994,9621,9453,8260,9625,9315,9058,9512,8906,9985,8566,8185,8182,7993,8588,8936,8681,8930,7995,8581,8138,8583,9047,8832,9953,9615,8365,9271,8366,8071,8919,9104,8072,9216,8076,8821,8409,9184,9948,9752,9180,9143,8101,9800,9591,8521,8406,8270,9226,8946,9150,9493,8370,9965,9066,9940,9301,9278,9198,9149,9891,9892,8683,8842,9497,9895,8770,8929,9796,9601,9602,9513,9355,8164,9339,9113,8168,9442,8777,9762,9193,8551,9661,8311,8554,8555,8243,8240,9208,8318,9922,8002,8797,9290,8850,9740,8857,9267,9383,8995,9714,8913,8859,8483,8484,8917,8486,8762,9696,8647,9482,8818,8642,9976,9927};
            vecTrain[1] = {1304,217,769,1300,210,668,762,763,1260,663,219,133,132,130,135,134,1619,1611,491,1775,693,1544,696,669,1773,1079,1565,1073,1548,1374,344,819,939,547,544,400,810,1115,1676,549,814,935,408,1095,1315,1314,981,1312,719,868,717,675,1157,714,713,1190,1345,121,123,124,575,1653,268,1401,1017,59,1828,1825,1361,57,1512,50,1823,1392,537,415,1187,1721,297,827,361,418,1184,926,538,987,1081,704,194,311,310,777,315,192,270,392,390,1622,1666,113,69,279,81,87,1254,1418,794,836,837,834,378,1170,689,524,526,1252,1400,1787,369,1403,914,423,916,1337,426,1234,826,583,580,441,1668,446,1135,1330,1175,244,382,1473,241,1370,1709,242,249,1386,1640,1298,903,1434,787,1636,841,783,33,924,37,247,849,34,1521,1535,1534,1244,1531,439,436,648,1005,1004,432,1002,1000,1783,1087,621,1329,1222,1692,1220,573,334,629,1443,576,333,1665,178,177,176,256,1729,558,1151,289,1413,973,885,971,970,1582,1428,859,1737,731,181,736,1712,502,1728,659,1212,752,1359,633,469,468,1674,638,609,562,1350,1697,1287,566,567,1589,1597,1394,1730,169,1592,227,1088,1599,222,221,96,964,10,813,15,1643,1039,863,1377,867,1526,1105,1102,1451,720,765,817,1741,1108,729,605,1165,153,152,1160,154,746,788,1623,551,1648,1717,1720,1064,1216,1439,756,1051,1798,398,49,1191,1059,44,992,42,1013,957,871,1696,1446,1113,1444,326,1114,1119,353,1755,1291,1296,1297,688,1620,618,1442,205,1663,772,208,646,616,1277,511,1319,489,1232,486,1354,1349,481,1609,1043,1529,1041,1570,1046,1045,262,1201,945,1714,357,1198,1579,1471,476,477,350,1762,1478,686,681,1193,359};
            vecTrain[2] = {2964,2967,2658,2961,2717,3791,2962,3100,3392,2650,2657,3612,3765,3417,3571,3454,3325,3411,3656,3650,2098,3652,1830,2188,3483,1837,2523,2484,3253,3254,1944,3558,3618,3835,2157,2150,3177,3779,2920,2244,3180,3529,2010,3858,2243,3488,2466,2913,2387,3481,3755,3111,2469,2857,3718,3831,2749,3402,3337,3336,3133,3566,3338,1958,3563,2527,3038,2556,3565,3774,2493,3662,2994,2494,3032,3244,2147,2394,2193,3497,2259,3859,2252,2399,2476,2474,2473,2906,2872,3030,2909,3173,3667,2478,1921,2607,1924,3308,3295,1929,3670,3442,2820,3296,3300,2823,2309,3668,2540,3281,2663,2545,3028,2301,1851,3025,2548,2789,2986,3388,2306,2079,3842,2177,2174,2386,2178,3680,3818,2753,2604,3582,2441,3816,2141,3814,2939,2763,2448,2760,2284,1948,2931,3319,3289,1931,2320,3156,2839,1934,2837,3210,3158,3729,2831,3552,3390,2277,3019,2270,2577,1840,1847,2357,2378,2168,3094,3365,2375,2371,3759,2595,2615,3367,3613,2454,3753,3036,2297,2862,3802,2190,1875,3124,3310,3166,2982,1904,1905,2806,3700,2804,3822,2200,2201,3144,2052,2055,2054,3212,3364,2560,2888,2562,3362,1989,3134,3637,3852,2364,3081,2417,2682,3082,2587,2239,2585,2621,2624,2421,2818,3071,1863,1862,3728,2186,2185,1866,1911,1910,1913,1912,3720,3721,2952,1916,3206,3205,2216,3277,3425,3374,3847,3208,2024,2757,3160,2027,2752,2021,2750,3013,3635,2515,2695,3458,2510,2106,2104,2736,2864,2814,2635,3221,3634,2430,3632,2739,3380,2816,3535,1971,3149,2236,1890,1891,2708,2136,2134,3231,3823,3510,2343,3235,2943,3238,3341,2904,1961,3689,3105,3186,3394,2745,2744,3009,2507,2503,3189,2501,2674,2649,2725,2724,2722,3781,3621,3060,3230,2729,2988,1887,1886,1885,3706,3704,3472,2121,2120,3780,2043,3554,2783,2126,3598,2080,3507,2235,2234,2084,2085,2933,3194,3191,2005,2532,2009,3199,1864};
            vecTrain[3] = {4303,5172,4726,4307,3924,4304,5847,5094,5316,4479,4039,4572,5475,5744,4229,5221,4607,4475,5077,5590,4029,5561,5988,4621,5582,5629,4678,5375,4676,4675,5624,5373,4670,4868,5808,5907,4989,4863,5516,5663,5065,3938,4931,5144,5017,4330,4031,5327,5996,4566,5467,4237,6000,4232,5187,5267,5476,5161,4847,3883,5330,4404,5301,4738,5596,5304,4135,4644,5307,4484,5612,5856,5857,5919,5918,5695,4975,5967,4800,4489,5811,5011,4878,5895,4092,4892,3943,3944,4401,4208,4656,5928,4200,4659,4553,5760,4315,4282,5833,4129,5134,3957,5257,4419,5855,4725,5565,5387,4414,4048,4126,4723,4438,4497,4318,4219,5667,5989,4967,4965,5686,5964,5349,5687,5000,5567,3953,4197,5671,5008,4192,3955,4620,4117,5672,4115,4358,4190,4626,5488,4354,5125,5126,5715,4544,4547,5408,5689,5249,5434,5579,4050,5083,4294,4421,4426,4290,5409,4425,5529,5829,5419,5132,4827,5549,4957,5730,5825,4188,5981,5934,4267,5037,4183,5331,4970,4631,5013,4637,5334,4635,5664,4109,5826,4638,4342,4340,5548,4946,4529,4657,5941,5704,4742,5438,4825,4746,5950,5898,5783,5901,5780,5868,5527,5019,5865,5867,5294,5863,5976,5559,4272,5173,4279,4278,3874,4716,5026,4440,5554,4442,4443,4924,5553,5250,5773,4373,5654,4371,5106,5651,4839,4687,3887,4933,5955,3880,4172,4173,5717,4830,5188,4919,4071,4070,5225,5797,5796,5576,5212,5814,5115,4082,5440,4084,5963,3903,5392,5087,4244,5313,4368,3864,4453,5555,4763,5880,4765,4456,4766,4768,5889,5084,4615,5756,4691,4009,5888,4927,4162,4161,4160,5720,4511,5454,4517,4007,3972,4594,3979,4551,4840,5806,5214,4993,3917,4258,4312,4241,4096,4251,5280,4099,5540,5979,4928,3918,5735,5734,5531,5417,5204,5049,4804,5044,5738,5040,4398,4563,4150,4151,4710,4711,5572,4713,5310,4917,5636,4159,4910,5400,4808,5568,3989,4858,4857,4855,3980,3983};
            vecTrain[4] = {7586,6963,7631,6967,7606,6968,7652,7644,7611,7112,6721,6727,7205,7840,6122,6086,7570,6729,7248,6952,7959,6870,7456,7457,6479,6715,7643,6473,6645,6646,6476,6008,7613,6248,7455,7697,6322,7692,7193,6241,6244,7081,7803,7802,7805,7262,7269,7859,6991,7286,6915,7047,7046,6718,7566,6158,6717,6092,7049,6919,6710,6449,6011,6010,7473,6108,6659,7446,7607,6018,6391,6653,6397,7111,7522,7729,6316,7492,6564,6252,7254,6256,7224,6591,6654,7461,6850,6988,6599,6652,6709,6142,6141,7877,6383,6701,7130,7133,6705,7136,6518,6519,7639,7243,7382,6261,7360,7363,7636,6667,7953,7096,7889,7888,7737,7915,6062,6953,6301,7887,6058,6820,6821,6580,6587,6825,7710,6828,7899,6844,7301,6933,6930,7020,6936,6670,7024,7092,7029,7785,6172,6173,7272,6272,6778,7703,7700,6423,6697,6776,6500,7943,6429,7142,6773,7265,7425,7314,6766,6854,7712,7267,7544,7155,6376,7620,7576,6834,7524,7612,7152,6169,6601,6602,7851,7344,7458,7017,7858,6760,6920,6167,6539,6437,7517,7135,7603,6288,7180,6769,7856,6767,6285,6286,7559,6280,6281,6283,6042,6689,6206,7462,6201,7127,6362,7419,7557,6685,7725,7878,7923,7902,7299,6805,7688,6800,6801,6613,6887,6527,6526,6529,7148,6480,6618,6486,7488,6888,6485,6356,6357,7082,6450,6753,6110,6111,6291,7569,6196,6197,7675,6212,7762,7585,6055,7404,6198,7278,6218,7845,7935,7036,7285,7394,6811,7288,6498,7038,7177,6746,7857,6897,6942,7830,6898,6624,6940,7673,6600,6621,7571,7504,7209,7572,6551,7318,6346,7876,6414,7201,6558,6786,6413,6100,6222,7479,6185,7773,7836,6224,6225,6906,6023,7766,6026,7829,6972,7369,7176,6014,6730,6235,7234,7903,7232,7928,7107,7663,7662,7497,6634,7825,6547,7485,7335,7316,7322,7069,7067,7467,6462,6463,7665,6137,7339,7466,7188,7189,7469,6038,7746,7744,6033,7181,6237,6030,6338,6790,7143};
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
        assert(currdocinfo.m_docword.size() > 0);
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
    random_device rdp,rdn;
    mt19937 ren(rdn());
    long totalinstance = 0;
    if(m_DocCorp.size() <= 0 || m_QueryCorp.size() <= 0){
		  throw MyError(" Error: query corp or doc corp should be initialized!");
    }
    for(auto iter = m_dataset.begin(); iter != m_dataset.end(); ++ iter){
        QINDEX cqindex = iter->first;
        if(m_QueryCorp.find(cqindex) == m_QueryCorp.end() || m_QueryCorp[cqindex].size() <= 0)  continue;
        m_QInstance.insert(make_pair(cqindex,vector<pair<string,string>>()));
        map<double,vector<string>,std::greater<double>> currsamples;
        int num_pos_currquery = 0;
        for(auto viter = iter->second.begin(); viter != iter->second.end(); ++ viter){
            if(m_DocCorp.find(*viter) == m_DocCorp.end() || m_DocCorp[*viter].m_docword.size() <= 0) continue;
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
        //assert(m_relinfo.find(currqindex) != m_relinfo.end());
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
                        bool bvalidq = Simi_evaluate(1,qIndex,currscore,1000,20,std::ref(tmpeval));
                        currTestRes += tmpeval;
                        if(!m_CalAllQ && bvalidq){
                            ++iValidTestQuery;
                        }else{
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
            bool bvalidq = Simi_evaluate(1,qIndex,currscore,1000,20,std::ref(tmpeval));
            cTestRes += tmpeval;
            if(!m_CalAllQ && bvalidq){
                ++iValidTestQuery;
            }else{
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
    if(iDocLen == 0 || nQSize == 0){
      return std::numeric_limits<double>::lowest();
    }
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
