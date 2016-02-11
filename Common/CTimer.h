/******************************************************************************
                         .88888.   888888ba  dP     dP 
                        d8'   `88  88    `8b 88     88 
                        88        a88aaaa8P' 88     88 
                        88   YP88  88        88     88 
                        Y8.   .88  88        Y8.   .8P 
                         `88888'   dP        `Y88888P' 
                                                       
                                                       
   a88888b.                                         dP   oo                   
  d8'   `88                                         88                        
  88        .d8888b. 88d8b.d8b. 88d888b. dP    dP d8888P dP 88d888b. .d8888b. 
  88        88'  `88 88'`88'`88 88'  `88 88    88   88   88 88'  `88 88'  `88 
  Y8.   .88 88.  .88 88  88  88 88.  .88 88.  .88   88   88 88    88 88.  .88 
   Y88888P' `88888P' dP  dP  dP 88Y888P' `88888P'   dP   dP dP    dP `8888P88 
                                88                                        .88 
                                dP                                    d8888P  
******************************************************************************/

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
