#ifndef	_CONFIG_H_
#define	_CONFIG_H_

#ifdef _WIN32
#include<Windows.h>
#else
#include<unistd.h>
#endif
#include<stdio.h>
#include<time.h>
#include<cstring>
#include<string>
#include<iostream>
#include<fstream>
#include<map>
#include<stdexcept>

namespace nsnn4ir{
#define MAX_PATH_LEN 512
	class MyError:public std::runtime_error{
	public:
		MyError(const std::string &msg=""):runtime_error(msg){}
	};

	class Config{
	protected:
		std::map<std::string,std::string> m_Contents;
		typedef std::map<std::string,std::string>::iterator mapiter;
		typedef std::map<std::string,std::string>::const_iterator mapciter;
	public:
		static Config& GetConfigInstance();
        void SetConfigFile(const std::string &sfilename,const std::string &delimiter="=",const std::string &comment="#");
		static void Trim(std::string &inout_s);
		int iValue(const std::string &skey)const;
		long lValue(const std::string &skey)const;
		float fValue(const std::string &skey)const;
		bool bValue(const std::string &skey)const;
		std::string sValue(const std::string &skey)const;
	private:
		Config(const std::string &sfilename="./config.ini",const std::string &delimiter="=",const std::string &comment="#");
		struct Object_Creat{
			Object_Creat(){
				Config::GetConfigInstance();
			}
		};
		static Object_Creat objector;
		Config(const Config&);
		Config& operator=(const Config&);
	};
}
#endif
