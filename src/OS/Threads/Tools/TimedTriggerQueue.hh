// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_TIMEDTRIGGERQUEUE_HEADER
#define RAVL_TIMEDTRIGGERQUEUE_HEADER 1
/////////////////////////////////////////////////////////
//! rcsid="$Id$"
//! file="Ravl/OS/Threads/Tools/TimedTriggerQueue.hh"
//! lib=RavlThreads
//! docentry="Ravl.API.OS.Time"
//! author="Charles Galambos"
//! date="23/09/1999"

#include "Ravl/OS/Date.hh"
#include "Ravl/PriQueue.hh"
#include "Ravl/Hash.hh"
#include "Ravl/RCLayer.hh"
#include "Ravl/Threads/Mutex.hh"
#include "Ravl/Threads/Semaphore.hh"
#include "Ravl/Threads/ThreadEvent.hh"
#include "Ravl/Calls.hh"
#include "Ravl/Service.hh"

// The new timed trigger code doesn't rely the un*x select system call,
// but gives less accurate timing.  

namespace RavlN
{
  class TimedTriggerQueueC;
  
  //! userlevel=Develop
  //: Timed event queue body.
  // This is a queue of routines to be triggered at some point in the future.
  // See the handle class for more details.
  
  class TimedTriggerQueueBodyC 
    : public ServiceC
  {
  public:
    TimedTriggerQueueBodyC(const XMLFactoryContextC &factory);
    //: Default constructor

    TimedTriggerQueueBodyC();
    //: Default constructor

    ~TimedTriggerQueueBodyC();
    //: Destructor.
    // This will not return until shutdown is complete.
    
    UIntT Schedule(RealT t,const TriggerC &se,float period = -1);
    //: Schedule event for running after time 't' (in seconds).
    // Thread safe.
    // Returns an ID for the event, which can
    // be used for cancelling it.
    
    UIntT Schedule(const DateC &startAt,const TriggerC &se,float period = -1);
    //: Schedule event for running.
    // Thread safe.
    // Returns an ID for the event, which can
    // be used for cancelling it.

    UIntT SchedulePeriodic(const TriggerC &se,float period);
    //: Schedule event for running periodically.
    // Thread safe.
    // Returns an ID for the event, which can
    // be used for cancelling it.

    bool Cancel(UIntT eventID);
    //: Cancel pending event.
    // Will return TRUE if event in cancelled before
    // it was run.

    virtual bool Shutdown();
    //: Shutdown even queue.
    
    virtual bool Start();
    //: Start service.

    typedef RavlN::SmartOwnerPtrC<TimedTriggerQueueBodyC> RefT;
    //: Owner reference counted ptr to class

    typedef RavlN::SmartCallbackPtrC<TimedTriggerQueueBodyC> CBRefT;
    //: Callback reference counter ptr to class

  protected:
    bool Process();
    //: Process event queue.

    virtual void ZeroOwners();
    //: Called when owning handles drops to zero.
    
    MutexC access;
    bool m_started;
    UIntT eventCount;
    PriQueueC<DateC,UIntT> schedule;
    HashC<UIntT,Tuple2C<TriggerC,float> > events;
    bool done;
    // Queue fd's
    SemaphoreC semaSched;
    
    friend class TimedTriggerQueueC;
  };
  
  //! userlevel=Normal
  //: Timed event queue.
  // This is a queue of routines to be triggered at some point in the future.
  // This class creates a new thread which triggers a list of routines
  // in the requested sequence.  A single thread is used to call all the
  // routines, so if any lengthy processing is required it is better to
  // spawn a new thread to complete it.   
  
  class TimedTriggerQueueC 
    : public RCLayerC<TimedTriggerQueueBodyC>
  {
  public:
    TimedTriggerQueueC()
    {}
    //: Default constructor.
    // Creates an invalid handle.

    TimedTriggerQueueC(bool )
      : RCLayerC<TimedTriggerQueueBodyC>(*new TimedTriggerQueueBodyC())
    {}
    //: Default constructor.
    // Creates an invalid handle.
    
  protected:
    TimedTriggerQueueC(TimedTriggerQueueBodyC &bod,RCLayerHandleT handleType = RCLH_OWNER)
      : RCLayerC<TimedTriggerQueueBodyC>(bod,handleType)
    {}
    //: Body constructor.
    
    TimedTriggerQueueBodyC &Body() 
    { return RCLayerC<TimedTriggerQueueBodyC>::Body(); }
    //: Access body.

    const TimedTriggerQueueBodyC &Body() const
    { return RCLayerC<TimedTriggerQueueBodyC>::Body(); }
    //: Access body.
    
    bool Process()
    { return Body().Process(); }
    //: Used to start internal thread.
  public:
    UIntT Schedule(RealT t,const TriggerC &se,float period = -1)
    { return Body().Schedule(t,se,period); }
    //: Schedule event for running after time 't' (in seconds).
    // Thread safe.
    // Returns an ID for the event, which can
    // be used for cancelling it.
    
    UIntT Schedule(const DateC &at,const TriggerC &se,float period = -1)
    { return Body().Schedule(at,se,period); }
    //: Schedule event for running.
    // Thread safe.
    // Returns an ID for the event, which can
    // be used for cancelling it.
    
    UIntT SchedulePeriodic(const TriggerC &se,float period)
    { return Body().SchedulePeriodic(se,period); }
    //: Schedule event for running periodically.
    // Thread safe.
    // Returns an ID for the event, which can
    // be used for cancelling it.

    bool Cancel(UIntT eventID)
    { return Body().Cancel(eventID); }
    //: Cancel pending event.
    // Will return TRUE if event in cancelled before
    // it was run.
    
    void Shutdown()
    { Body().Shutdown(); }
    //: Shutdown even queue.
    
    friend class TimedTriggerQueueBodyC;
  };

  //! Access a global trigger queue.
  // As one thread is sharing all the work,
  // long (>0.1s) tasks be spawned on a separate thread.
  TimedTriggerQueueC GlobalTriggerQueue();
  
}

#endif
