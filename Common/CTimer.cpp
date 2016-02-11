/******************************************************************************
GPU Computing / GPGPU Praktikum source code.

******************************************************************************/

#include "CTimer.h"

using namespace std;
using namespace std::chrono;

///////////////////////////////////////////////////////////////////////////////
// CTimer

void CTimer::Start()
{
	m_StartTime = Clock::now();
}

void CTimer::Stop()
{
	m_EndTime = Clock::now();
}

double CTimer::GetElapsedMilliseconds()
{
	return duration_cast<nanoseconds>(m_EndTime - m_StartTime).count() * 1e-6;
}

///////////////////////////////////////////////////////////////////////////////
