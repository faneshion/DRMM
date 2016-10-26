/*
 * test
*/

#include<stdio.h>

#include "NN4IR.h"
#include "Config.h"

//using namespace nsnn4ir;

void test();
int ArgPos(char *str, int argc, char **argv);

int main(int argc,char * argv[]){
    omp_set_num_threads(23);
    omp_init_lock(&lock);
    int i = -1;
    char configfile[50];
    if ((i = ArgPos((char *)"-config", argc, argv)) > 0) strcpy(configfile,argv[i + 1]);
    if(strlen(configfile) > 0)  nsnn4ir::Config::GetConfigInstance().SetConfigFile(configfile, "=", "#");
    long corpus_doc_count = nsnn4ir::Config::GetConfigInstance().lValue("CORPUS_DOC_COUNT");
    double lr_w1 = nsnn4ir::Config::GetConfigInstance().fValue("LR_W1");
    double lr_w2 = nsnn4ir::Config::GetConfigInstance().fValue("LR_W2");
    int mini_batch = nsnn4ir::Config::GetConfigInstance().iValue("MINI_BATCH");
    int fold_size = nsnn4ir::Config::GetConfigInstance().iValue("FOLD_SIZE");
    string corpus_term_dfcf_file = nsnn4ir::Config::GetConfigInstance().sValue("CORPUS_TERM_DFCF_FILE");
    string query_data_file = nsnn4ir::Config::GetConfigInstance().sValue("QUERY_DATA_FILE");
    string doc_data_file = nsnn4ir::Config::GetConfigInstance().sValue("DOC_DATA_FILE");
    string rerank_data_file = nsnn4ir::Config::GetConfigInstance().sValue("RERANK_DATA_FILE");
    string qrel_file = nsnn4ir::Config::GetConfigInstance().sValue("QREL_FILE");
    string qrel_idcg_file = nsnn4ir::Config::GetConfigInstance().sValue("QREL_IDCG_FILE");
    string word_embed_file = nsnn4ir::Config::GetConfigInstance().sValue("WORD_EMBED_FILE");
    string save_ranklist_file = nsnn4ir::Config::GetConfigInstance().sValue("SAVE_RANKLIST_FILE");
    nsnn4ir::_enDataType task_type =nsnn4ir:: _enDataType(nsnn4ir::Config::GetConfigInstance().iValue("TASK_TYPE"));
    long sample_total_limited = nsnn4ir::Config::GetConfigInstance().lValue("SAMPLE_TOTAL_LIMITED");
    int sample_perpositive_limited = nsnn4ir::Config::GetConfigInstance().iValue("SAMPLE_PERPOSITIVE_LIMITED");
    int sample_perquery_limited = nsnn4ir::Config::GetConfigInstance().iValue("SAMPLE_PERQUERY_LIMITED");
    int max_iteration = nsnn4ir::Config::GetConfigInstance().iValue("MAX_ITERATION");
    bool cal_all_q = nsnn4ir::Config::GetConfigInstance().bValue("CAL_ALL_Q");
    nsnn4ir::_enActivationType activation_func_type = nsnn4ir::_enActivationType(nsnn4ir::Config::GetConfigInstance().iValue("ACTIVATION_FUNC_TYPE"));

    nsnn4ir::NN4IR  * pweor = new nsnn4ir::NN4IR(lr_w1, lr_w2, mini_batch, activation_func_type, cal_all_q);
    pweor->setDataSet(task_type);
    pweor->InitGroundTruth(qrel_file,qrel_idcg_file,1);
    pweor->InitCorpInfo(corpus_term_dfcf_file,corpus_doc_count);
    pweor->InitQueryCorp(query_data_file);
    pweor->InitDocCorp(doc_data_file);
    pweor->LoadDataSet(rerank_data_file,sample_total_limited,sample_perpositive_limited,sample_perquery_limited);
    pweor->InitWordVec(word_embed_file,true); // initial word embedding 
    pweor->InitTopKNeiB(); // calculate word similarity in advance

    pweor->RunningMultiThread(fold_size,max_iteration); //simi
    pweor->GetRanklist(save_ranklist_file.c_str());
    delete pweor;
    pweor = NULL;
    cout<<"Done ....\n";
    return 0;
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}
