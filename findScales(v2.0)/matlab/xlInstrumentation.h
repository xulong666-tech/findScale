#pragma once
/******************************************************** 
	* @file    : xlInstrumentation.h
	* @brief   : this file is used to visualize the time consume of fcuntion you want to see.An example is put at the end.
	* @details : '#define PROFILING 1' to using this file.the result file is stored as 'result.json' ,type in 'chrome://tracing'
					and drag 'result.json' into Chrome you we see the visual result.
	* @author  : longxu@nuaa.edu.cn
	* @date    : 2023-6-19
*********************************************************/

#include <iostream>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <chrono>//use this library to  calculate time consume of algorithm
#include <thread>
#define PROFILING 1
#if PROFILING

#define PROFILE_SCOPE(name) InstrumentationTimer timer##__LINE__(name)

#else
#define  PROFILE_SCOPE(name)
#endif

//#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCSIG__)
struct ProfileResult
{
	std::string Name;
	long long Start, End;
	uint32_t ThreadID;
};
struct InstrumentationSession
{
	std::string Name;
};
class Instrumentor
{

private:
	InstrumentationSession* m_CurrentSession;
	std::ofstream m_OutputStream;
	int m_ProfileCount;
public:
	Instrumentor()
		:m_CurrentSession(nullptr), m_ProfileCount(0)
	{
	}
	void BeginSession(const std::string& name, const std::string& filepath = "results.json")
	{
		m_OutputStream.open(filepath);
		WriteHeader();
		m_CurrentSession = new InstrumentationSession{ name };

	}
	void EndSession()
	{
		WriteFooter();
		m_OutputStream.close();
		delete m_CurrentSession;
		m_CurrentSession = nullptr;
		m_ProfileCount = 0;
	}
	void WriteProfile(const ProfileResult& result)
	{
		if (m_ProfileCount++ > 0)
			m_OutputStream << ",";

		std::string name = result.Name;
		std::replace(name.begin(), name.end(), '"', '\'');

		m_OutputStream << "{";
		m_OutputStream << "\"cat\":\"function\",";
		m_OutputStream << "\"dur\":" << (result.End - result.Start) << ',';
		m_OutputStream << "\"name\":\"" << name << "\",";
		m_OutputStream << "\"ph\":\"X\",";
		m_OutputStream << "\"pid\":0,";
		m_OutputStream << "\"tid\":"<<result.ThreadID<<",";
		m_OutputStream << "\"ts\":" << result.Start;
		m_OutputStream << "}";

		m_OutputStream.flush();
	}
	void WriteHeader()
	{
		m_OutputStream << "{\"otherData\":{},\"traceEvents\":[";
		m_OutputStream.flush();

	}
	void WriteFooter()
	{
		m_OutputStream << "]}";
		m_OutputStream.flush();

	}
	static Instrumentor& Get()
	{
		static Instrumentor* instance = new Instrumentor();
		return *instance;
	}
};
class InstrumentationTimer
{
public:
	InstrumentationTimer(const char* name)
		:m_Name(name), m_Stopped(false)
	{
		m_StartTimepoint = std::chrono::high_resolution_clock::now();
	}

	~InstrumentationTimer()
	{
		if (!m_Stopped)
			Stop();
	}
	void Stop()
	{
		auto endTimepoint = std::chrono::high_resolution_clock::now();

		long long start = std::chrono::time_point_cast<std::chrono::microseconds>(m_StartTimepoint).time_since_epoch().count();
		long long end = std::chrono::time_point_cast<std::chrono::microseconds>(endTimepoint).time_since_epoch().count();

		std::cout << m_Name << ": " << (end - start) << "us\n";
		uint32_t threadID = std::hash<std::thread::id>{}(std::this_thread::get_id());
		Instrumentor::Get().WriteProfile({ m_Name,start,end,threadID });
		m_Stopped = true;
	}
private:
	const char* m_Name;
	std::chrono::time_point<std::chrono::steady_clock> m_StartTimepoint;
	bool m_Stopped;
};
/**********************************************example******************************************* :
#include <iostream>
#include <string>
#include <string_view>
#include "xlInstrumentation.h"

void func1()
{
	PROFILE_FUNCTION();

	for (int i = 0; i < 1000; i++)
	{
		std::cout << "Hello word1:" << std::endl;

	}
}
void func2()
{
	PROFILE_FUNCTION();

	for (int i = 0; i < 1000; i++)
	{
		std::cout << "Hello word2:" << std::endl;

	}
}
void RunBenchmark()
{
	PROFILE_FUNCTION();

	func1();
	func2();
}
int main()
{
	Instrumentor::Get().BeginSession("Profile");
	RunBenchmark();
	Instrumentor::Get().EndSession();

	std::cin.get();
}
*/

