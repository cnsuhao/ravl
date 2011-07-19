// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLTHREADS_CONDITIONAL_HEADER
#define RAVLTHREADS_CONDITIONAL_HEADER 1
/////////////////////////////////////////////////
//! rcsid="$Id$"
//! file="Ravl/OS/Threads/Posix/ConditionalMutex.hh"
//! lib=RavlThreads
//! userlevel=Normal
//! docentry="Ravl.API.OS.Threads"
//! author="Charles Galambos"
//! date="02/07/1999"

#include "Ravl/config.h"

#if !defined(_POSIX_SOURCE) && !defined(__sgi__) && !RAVL_OS_FREEBSD
#define _POSIX_SOURCE 1
#endif

//#if defined(__sol2__)
#if RAVL_HAVE_SIGNAL_H
#include <sys/signal.h>
#endif

#if RAVL_HAVE_WIN32_THREADS
#include <windows.h>
#endif
#if RAVL_HAVE_POSIX_THREADS
#include <pthread.h>
#endif

#include "Ravl/Types.hh"
#include "Ravl/Threads/Mutex.hh"

namespace RavlN
{
  
  class DateC;

  //! userlevel=Normal
  //: Sleeping until signal arrives.
  //
  // <p>This class implements a "condition variable".  
  // It causes a thread to sleep until signalled from another thread.  </p>
  //
  // <p>ConditionalMutexC wraps the pthreads condition variable and
  // its associated mutex
  // into a single object.  See man pages on e.g. pthread_cond_init for
  // a full description.  See also SemaphoreC for an example of its
  // use.</p>
  //
  // <p>In this class, Wait() will only wake up once after each Broadcast(), which is why it does not need resetting after a Broadcast() (in contrast to <a href="RavlN.ThreadEventC.html">ThreadEventC</a>).  If this "edge-triggered" behaviour is not what is wanted, or if this class is used only for its ability to wake up other sleeping threads, <a href="RavlN.ThreadEventC.html">ThreadEventC</a> may be a better choice.</p>
  
  class ConditionalMutexC 
    : public MutexC 
  {
  public:
    ConditionalMutexC() 
#if RAVL_HAVE_PTHREAD_COND 
    { if(pthread_cond_init(&cond,0)) 
      Error("pthread_cond_init failed. \n");
    }
#else
    ;
#endif
    //: Constructor.
    
    ~ConditionalMutexC();
    //: Destructor

    void Broadcast()
#if RAVL_HAVE_PTHREAD_COND 
    { pthread_cond_broadcast(&cond); }
    //: Broadcast a signal to all waiting threads.
    // Always succeeds.
#else
    ;
#endif
    
    void Signal() 
#if RAVL_HAVE_PTHREAD_COND 
    { pthread_cond_signal(&cond); }
    //: Signal one waiting thread.
    // Always succeeds.  The particular thread selected is arbitrary.
#else
    ;
#endif
    
    void Wait()
#if RAVL_HAVE_PTHREAD_COND 
    { pthread_cond_wait(&cond,&mutex); }
    //: Wait for conditional.
    // <p>This unlocks the mutex and then waits for a signal
    // from either Signal or Broadcast.  When it gets the signal
    // the mutex is re-locked and control returned to the
    // program. </p>
    // <p>Always succeeds.</p>
#else
    ;
#endif
    
    bool Wait(RealT maxTime);
    //: Wait for conditional.
    // This unlocks the mutex and then waits for a signal
    // from either Signal, Broadcast or timeout.  When it get the signal
    // the mutex is re-locked and control returned to the
    // program. <p>
    // Returns false, if timeout occurs.

    bool WaitUntil(const DateC &deadline);
    //: Wait for conditional.
    // This unlocks the mutex and then waits for a signal
    // from either Signal, Broadcast or timeout.  When it get the signal
    // the mutex is re-locked and control returned to the
    // program. <p>
    // Returns false, if timeout occurs.


  private:
    ConditionalMutexC(const ConditionalMutexC &)
    {}
    //: This is just a bad idea.
    
#if RAVL_HAVE_PTHREAD_COND 
    pthread_cond_t cond;
#endif
#if RAVL_HAVE_WIN32_THREADS
    volatile LONG count;
    HANDLE sema; // Flow control semaphore.
#endif
  };
}

#endif
