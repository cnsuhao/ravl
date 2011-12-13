#ifndef RAVL_THREADLIMITCOUNTER_HH_
#define RAVL_THREADLIMITCOUNTER_HH_
/////////////////////////////////////////////////
//! file="Ravl/OS/Threads/Posix/ThreadLimitCounter.hh"
//! lib=RavlThreads
//! userlevel=Normal
//! docentry="Ravl.API.OS.Threads"
//! author="Charles Galambos"
//! date="25/05/2011"

#include "Ravl/config.h"
#include "Ravl/Threads/ConditionalVariable.hh"
#include "Ravl/Stream.hh"
#include "Ravl/Threads/Signal1.hh"
#include "Ravl/OS/Date.hh"

namespace RavlN
{


  //! userlevel=Normal
  //: Thread safe limit counter.
  // This behaves much like a semaphore, but allows increments and decrements of more than 1.

  class ThreadLimitCounterC
  {
  public:
    ThreadLimitCounterC(UIntT count);
    //: Constructor

    void Release(UIntT units);
    //: Release a number of units back to the pool.

    bool Request(UIntT units);
    //: Request a number of units.
    //: Note: With a stream of many small requests, there
    //: is a chance a large request will never be satisfied.

    bool RequestWait(UIntT units,float maxTime);
    //: Request a number of units, with a timeout
    //: Note: With a stream of many small requests, there
    //: is a chance a large request will never be satisfied.

    bool WaitForFree() const;
    //: Wait for lock to be free of all waiters.

    UIntT ThreadsWaiting() const
    { return m_waiting; }
    //: Get approximation of number of threads waiting.

  protected:
    IntT m_count;
    IntT m_maxCount;
    mutable MutexC m_access;
    mutable ConditionalVariableC m_cond;
    volatile UIntT m_waiting; // Count of number of threads waiting on this...
  };
};


#endif /* RAVL_THREADLIMITCOUNTER_HH_ */
