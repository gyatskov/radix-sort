#ifndef _CTIMER_H
#define _CTIMER_H

#include <chrono>

//! Simple wrapper class for the measurement of time intervals
/*!
	Use this timer to measure elapsed time on the HOST side.
	Not suitable for measuring the execution of DEVICE code
	without synchronization with the HOST.

	NOTE: This class is not thread-safe (like most other classes in these
	examples), but we are not doing CPU multithreading in the praktikum...
*/
class CTimer
{
public:

	typedef std::chrono::high_resolution_clock		Clock;
	typedef Clock::time_point						TimePoint;

	CTimer(){};

	~CTimer(){};

	void Start();

	void Stop();

	//! Returns the elapsed time between Start() and Stop() in ms.
	double GetElapsedMilliseconds();

protected:
	TimePoint			m_StartTime;
	TimePoint			m_EndTime;
};

#endif // _CTIMER_H
