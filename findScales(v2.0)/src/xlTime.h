#pragma once
#include "xlTime.h"
#include <iostream>
#include <string>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <chrono>//use this library to  calculate time consume of algorithm

namespace xl
{
	//used to calculate the time consume of a algorithm
	#define CoutTime(func)	{\
	xl::Timer time;\
	func;	\
	std::cout<< std::endl<< #func <<std::endl;\
	}\

	struct Timer
	{
		std::chrono::time_point<std::chrono::steady_clock> start, end;
		std::chrono::duration<float> duration;

		Timer()
		{
			start = std::chrono::high_resolution_clock::now();
		}
		~Timer()
		{
			end = std::chrono::high_resolution_clock::now();
			duration = end - start;
			float ms = duration.count() *1000.0f;
			std::cout << "Timer took:" << ms << "ms" << std::endl;
		}
	};
}
