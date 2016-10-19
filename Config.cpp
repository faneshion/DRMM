
#include "Config.h"

namespace nsnn4ir{
	Config::Object_Creat Config::objector;
	Config& Config::GetConfigInstance(){
		static Config _AttrawlConfigdata;
		return _AttrawlConfigdata;
	}
	Config::Config(const std::string & filename,const std::string &delimiter,const std::string &comment){
        SetConfigFile(filename, delimiter, comment);
	}
    void Config::SetConfigFile(const std::string & filename,const std::string &delimiter,const std::string &comment){
		std::ifstream in(filename.c_str());
		if(!in)
			throw MyError("Error: config file not exist!");
		std::string sline;
		std::string nextline = "";// might need to read ahead to see where value ends
		while(in || nextline.length() > 0){
			std::string sline;
			if(nextline.length() > 0){
				sline = nextline;
				nextline = "";
			}else{
				std::getline(in,sline);
			}
			sline = sline.substr(0,sline.find(comment));
			size_t delimpos = sline.find(delimiter);
			if(delimpos < std::string::npos){
				std::string skey = sline.substr(0,delimpos);
				sline.replace(0,delimpos+delimiter.length(),"");
				bool terminate = false;
				while(!terminate && in){
					std::getline(in,nextline);
					terminate = true;
					std::string nlcopy = nextline;
					Config::Trim(nlcopy);
					if(nlcopy == "")
						continue;
					nextline = nextline.substr(0,nextline.find(comment));
					if(nextline.find(delimiter) != std::string::npos)
						continue;
					nlcopy = nextline;
					Config::Trim(nlcopy);
					if(nlcopy != "") sline += "\n";
					sline +=nextline;
					terminate =false;
				}
				Config::Trim(skey);
				Config::Trim(sline);
				m_Contents[skey] = sline;
			}
		}
    }
	int Config::iValue(const std::string &skey)const{
		mapciter iter = m_Contents.find(skey);
		if(iter == m_Contents.end())
			throw MyError("Error: Config:Key "+skey+" Not Exist");
		return atoi(iter->second.c_str());
	}
	long Config::lValue(const std::string &skey)const{
		mapciter iter = m_Contents.find(skey);
		if(iter == m_Contents.end())
			throw MyError("Error: Config:Key "+skey+" Not Exist");
		return atol(iter->second.c_str());
	}
	float Config::fValue(const std::string &skey)const{
		mapciter iter = m_Contents.find(skey);
		if(iter == m_Contents.end())
			throw MyError("Error: Config:Key "+skey+" Not Exist");
		return atof(iter->second.c_str());
	}
	bool Config::bValue(const std::string &skey)const{
		mapciter iter = m_Contents.find(skey);
		if(iter == m_Contents.end())
			throw MyError("Error: Config:Key "+skey+" Not Exist");
		return atoi(iter->second.c_str());
	}
	std::string Config::sValue(const std::string &skey)const{
		mapciter iter = m_Contents.find(skey);
		if(iter == m_Contents.end())
			throw MyError("Error: Config:Key "+skey+" Not Exist");
		return iter->second;
	}
	void Config::Trim(std::string &inout_s){
		static const char whitespace[] = " \n\t\v\r\f";
		inout_s.erase(0,inout_s.find_first_not_of(whitespace));
		inout_s.erase(inout_s.find_last_not_of(whitespace)+1U);
	}
}
