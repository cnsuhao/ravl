// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
////////////////////////////////////////////////////
//! rcsid="$Id$"
//! lib=RavlThreads
//! file="Ravl/OS/Threads/Tools/testTimedTriggerQueue.cc"
//! author="Charles Galambos"
//! docentry="Ravl.API.OS.Time"
//! userlevel=Develop

#include "Ravl/Threads/TimedTriggerQueue.hh"
#include "Ravl/Random.hh"
#include <math.h>

using namespace RavlN;

const int noTests = 10;
const RealT g_timePeriod = 2.0;
const RealT maxTimerError = 0.04;

DateC when[noTests];

bool MarkTime(int &i) {
  when[i] = DateC(true); // Mark now.
  RavlN::Sleep(0.05);
  //  cerr << " i:" << i << "\n";
  return true;
}


int main()
{
  cerr << "Running test... \n";
  TimedTriggerQueueC eventQueue(true);
  Sleep(0.01);  // Give event queue time to setup properly...
  DateC requestedTime[noTests];
  int i;
  for(i = 0;i < noTests;i++) {
#if 0
    // Test handling of no delay
    if(i % 2 == 0) {
      requestedTime[i] = DateC::NowUTC();
      eventQueue.Schedule(-1.0,Trigger(MarkTime,i));
    }
    else
#endif
    {
      requestedTime[i] = DateC::NowUTC() + Random1()*g_timePeriod;
      eventQueue.Schedule(requestedTime[i],Trigger(MarkTime,i));
    }
  }
  Sleep(g_timePeriod + maxTimerError*10.0);  // Wait for all events to take place.
  for(i = 0;i < noTests;i++) {
    RealT diff = (requestedTime[i] - when[i]).Double();
    cerr << "Timing error:" << diff << " " << requestedTime[i].ODBC(false,true) << " " << when[i].ODBC(false,true) << " \n";
    if(fabs(diff) > maxTimerError) {
      cerr << "ERROR: Timing out of spec \n";
      return 1;
    }
  }
  cerr << "TEST PASSED.\n";
  return 0; 
}

