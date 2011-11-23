// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
/////////////////////////////////////////////////////////
//! rcsid="$Id$"
//! lib=RavlThreads
//! file="Ravl/OS/Threads/Tools/TimedTriggerQueue.cc"

#include "Ravl/config.h"

#ifdef __sgi__
#undef _POSIX_C_SOURCE
#include <standards.h>
#endif

#include "Ravl/Threads/TimedTriggerQueue.hh"
#include "Ravl/Threads/LaunchThread.hh"
#include "Ravl/Exception.hh"

#define DODEBUG 0

#if DODEBUG 
#include "Ravl/String.hh"
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

// Disabling catching of exceptions can be useful
// if you need to establish what's throwing them
#define CATCHEXCEPTIONS 1

namespace RavlN
{

  //: Default constuctor
  
  TimedTriggerQueueBodyC::TimedTriggerQueueBodyC() 
    : eventCount(0),
      done(false),
      semaSched(0)
  {
    LaunchThread(TimedTriggerQueueC(*this,RCLH_CALLBACK),&TimedTriggerQueueC::Process);
  }
  
  //: Destructor.
  
  TimedTriggerQueueBodyC::~TimedTriggerQueueBodyC() {
  }
  
  //: Called when owning handles drops to zero.
  
  void TimedTriggerQueueBodyC::ZeroOwners() {
    // Start shutdown
    done = true;
    // Pass back to parent on principle.
    RCLayerBodyC::ZeroOwners();
  }
  
  //: Shutdown even queue.
  
  void TimedTriggerQueueBodyC::Shutdown() {
    done = true;
  }
  
  //: Schedule event for running after time 't' (in seconds).
  
  UIntT TimedTriggerQueueBodyC::Schedule(RealT t,const TriggerC &se,float period) {
    DateC at = DateC::NowUTC(); // Set to now.
    at += t; // Add delay...
    return Schedule(at,se,period);
  }

  //: Schedule event for running periodically.
  UIntT TimedTriggerQueueBodyC::SchedulePeriodic(const TriggerC &se,float period) {
    return Schedule(period,se,period);
  }

  //: Schedule event for running.
  // Thread safe.
  
  UIntT TimedTriggerQueueBodyC::Schedule(const DateC &at,const TriggerC &se,float period) {
    if(!se.IsValid())
      return 0;
    MutexLockC holdLock(access);
    do {
      eventCount++;
      if(eventCount == 0)
	      eventCount++;
    } while(events.IsElm(eventCount));
    UIntT nEvent = eventCount;
    schedule.Insert(at,nEvent);
    Tuple2C<TriggerC,float> &entry = events[nEvent];
    entry.Data1() = se;
    entry.Data2() = period; // Don't repeat
    
    holdLock.Unlock();
    ONDEBUG(std::cerr << "TimedTriggerQueueBodyC::Schedule() Event " << nEvent << " at " << at.Text() << " \n");
    semaSched.Post();
    return nEvent;
  }

  //: Process event queue.
  
  bool TimedTriggerQueueBodyC::Process() {
    ONDEBUG(std::cerr << "TimedTriggerQueueBodyC::Process(), Called. \n");
    MutexLockC holdLock(access);
    holdLock.Unlock();
#if CATCHEXCEPTIONS
    do {
      // Avoid try/catch in central loop to reduce overhead.
      try {
#endif
        do {
          holdLock.Lock();
          // Are any events scheduled ??
          if(!schedule.IsElm()) {
            ONDEBUG(std::cerr << "Waiting for event to be scheduled. Size:" << schedule.Size() << "\n");
            holdLock.Unlock();
            semaSched.Wait();
            ONDEBUG(std::cerr << "Re-checking event queue.\n");
            continue; // Go back and check...
          }
          DateC nextTime = schedule.TopKey();
          DateC now = DateC::NowUTC();
          DateC toGo = nextTime - now;
          ONDEBUG(std::cerr << "Next scheduled event in " << toGo.Double() << " seconds\n");
          if(toGo.Double() < 0.00001) { // Might as well do it now...
            int eventNo = schedule.GetTop();
            ONDEBUG(std::cerr << "Executing event  " << eventNo << "\n");
            Tuple2C<TriggerC,float> &entry = events[eventNo];
            TriggerC ne = entry.Data1();
            // Is function non-periodic or cancelled?
            if(entry.Data2() < 0 || !ne.IsValid()) {
              events.Del(eventNo); // Remove from queue.
            } else {
              // Reschedule periodic functions.
              DateC at = now + entry.Data2();
              schedule.Insert(at,eventNo);
            }
            holdLock.Unlock(); // Unlock before invoke the event, incase it wants to add another.
            if(ne.IsValid()) // Check if event has been canceled.
              ne.Invoke();
            ONDEBUG(else std::cerr << "Event cancelled. \n");
            continue; // Check for more pending events.
          }
          
          holdLock.Unlock();
          // Wait for delay, or until a new event in scheduled.
          semaSched.Wait(toGo.Double());
          ONDEBUG(std::cerr << "Time to check things out.\n");
        } while(!done);
#if CATCHEXCEPTIONS
      } catch(ExceptionC &ex) {
        std::cerr << "Caught exception in timed trigger event thread. Message:'" << ex.Text() << "' \n";
        // Dump a stack.
        ex.Dump(std::cerr);
        // If in check or debug stop.
        RavlAssertMsg(0,"Aborting due to exception in timed trigger event thread. ");
        // Otherwise ignore and struggle on.
        std::cerr << "Ignoring. \n";
      } catch(...) {
        // If in check or debug stop.
        RavlAssertMsg(0,"Caught exception in timed trigger event thread. ");
        // Otherwise ignore and struggle on.
        std::cerr << "Caught exception in timed trigger event thread. Ignoring. \n";
      }
    } while(!done) ;
#endif
    return true;
   }
  
  //: Cancel pending event.
  // Will return TRUE if event in canceled before
  // it was run.
  
  bool TimedTriggerQueueBodyC::Cancel(UIntT eventID) {
    MutexLockC holdLock(access);
    if(!events.IsElm(eventID))
      return false;
    events[eventID].Data1().Invalidate(); // Cancel event.
    return true;
  }

  //! Access a global trigger queue.
  // As one thread is sharing all the work,
  // long (>0.1s) tasks be spawned on a sperate thread.
  TimedTriggerQueueC GlobalTriggerQueue() {
    static TimedTriggerQueueC triggerQueue(true);
    return triggerQueue;
  }


}
