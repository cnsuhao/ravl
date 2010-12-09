// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
/////////////////////////////////////////
//! rcsid="$Id$"
//! lib=RavlThreads
//! file="Ravl/OS/Threads/Posix/RWLock.cc"

#include "Ravl/Threads/Thread.hh"
#include "Ravl/Threads/RWLock.hh"
#include "Ravl/Stream.hh"
#include "Ravl/OS/Date.hh"

#if defined(VISUAL_CPP)
#include <time.h>
#else
#include <sys/time.h>
#endif

#define NANOSEC 1000000000


namespace RavlN
{

#if RAVL_HAVE_POSIX_THREADS_RWLOCK
  //: Copy constructor.
  // This just creates another lock.
  
  RWLockC::RWLockC(const RWLockC &oth)
    : isValid(false),
      m_preferWriter(oth.m_preferWriter)
  {
    int ret;
#ifdef PTHREAD_RWLOCK_WRITER_NONRECURSIVE_INITIALIZER_NP
    if(m_preferWriter) {
      pthread_rwlockattr_t attr;
      pthread_rwlockattr_init(&attr);
      pthread_rwlockattr_setkind_np (&attr,PTHREAD_RWLOCK_PREFER_WRITER_NP);
      ret = pthread_rwlock_init(&id,&attr);
      pthread_rwlockattr_destroy (&attr);
    } else {
      ret = pthread_rwlock_init(&id,0);
    }
#else
    m_preferWriter = false; // Not supported.
    ret = pthread_rwlock_init(&id,0);
#endif
    if(ret != 0)
      Error("RWLockC::RWLockC, Init failed ",ret);
    else isValid = true;    
  }
  
  //: Constructor.
  
  RWLockC::RWLockC(bool preferWriter)
    : isValid(false),
      m_preferWriter(preferWriter)
  {
    int ret;
#ifdef PTHREAD_RWLOCK_WRITER_NONRECURSIVE_INITIALIZER_NP
    if(m_preferWriter) {
      pthread_rwlockattr_t attr;
      pthread_rwlockattr_init(&attr);
      pthread_rwlockattr_setkind_np (&attr,PTHREAD_RWLOCK_PREFER_WRITER_NP);
      ret = pthread_rwlock_init(&id,&attr);
      pthread_rwlockattr_destroy (&attr);
    } else {
      ret = pthread_rwlock_init(&id,0);
    }
#else
    m_preferWriter = false; // Not supported.
    ret = pthread_rwlock_init(&id,0);
#endif
    if(ret != 0)
      Error("RWLockC::RWLockC, Init failed ",ret);
    else isValid = true;
  }
  
  // Destructor.
  
  RWLockC::~RWLockC() { 
    int x = 100;
    // This could fail if lock is held.
    while(pthread_rwlock_destroy(&id) && x-- > 0)
      OSYield();
    isValid = false;
    if(x == 0) 
      cerr << "WARNING: Failed to destory RWLock. \n";
  }

  //: Get a read lock.

  bool RWLockC::RdLock(void)
  {
    IntT ret;
    errno = 0;
    RavlAssert(isValid);
    do {
      if((ret = pthread_rwlock_rdlock(&id)) == 0) {
        RavlAssert(isValid);
        return true;
      }
    } while(errno == EINTR || ret == EINTR);
    Error("Failed to get RdLock", ret);
    return false;
  }

  //: Aquire a read lock with timeout.
  
  bool RWLockC::RdLock(float timeout )
  {
    struct timespec ts;
    struct timeval tv;

    // Work out delay.
    long secs = Floor(timeout);
    long nsecs = (long) ((RealT) ((RealT) timeout - ((RealT) secs)) * NANOSEC);

    // Get current time.
    gettimeofday(&tv,0);
    ts.tv_sec = tv.tv_sec;
    ts.tv_nsec = tv.tv_usec * 1000;

    // Add them.
    ts.tv_sec += secs;
    ts.tv_nsec += nsecs;
    if(ts.tv_nsec >= NANOSEC) {
      ts.tv_sec += 1;
      ts.tv_nsec -= NANOSEC;
    }


    IntT ret;
    errno = 0;
    RavlAssert(isValid);
    do {
      if((ret = pthread_rwlock_timedrdlock(&id,&ts)) == 0) {
        RavlAssert(isValid);
        return true;
      }
    } while(errno == EINTR || ret == EINTR);
    RavlAssert(isValid);
    return false;
  }

  //: Aquire a write lock with timeout
  // Returns true if lock aquired, false if timeout.
  
  bool RWLockC::WrLock(float timeout) 
  {
    if(timeout < 0)
      return WrLock();
    
    struct timespec ts;
    struct timeval tv;

    // Work out delay.
    long secs = Floor(timeout);
    long nsecs = (long) ((RealT) ((RealT) timeout - ((RealT) secs)) * NANOSEC);

    // Get current time.
    gettimeofday(&tv,0);
    ts.tv_sec = tv.tv_sec;
    ts.tv_nsec = tv.tv_usec * 1000;

    // Add them.
    ts.tv_sec += secs;
    ts.tv_nsec += nsecs;
    if(ts.tv_nsec >= NANOSEC) {
      ts.tv_sec += 1;
      ts.tv_nsec -= NANOSEC;
    }


    IntT ret;
    errno = 0;
    RavlAssert(isValid);
    do {
      if((ret = pthread_rwlock_timedwrlock(&id,&ts)) == 0) {
        RavlAssert(isValid);
        return true;
      }
    } while(errno == EINTR || ret == EINTR);
    RavlAssert(isValid);
    return false;
  }

  //: Get a write lock.
  bool RWLockC::WrLock(void)
  {
    IntT ret;
    errno = 0;
    RavlAssert(isValid);
    do {
      if((ret = pthread_rwlock_wrlock(&id)) == 0) {
        RavlAssert(isValid);
        return true;
      }
    } while(errno == EINTR || ret == EINTR);
    RavlAssert(isValid);
    Error("Failed to get WrLock", ret);
    return false;
  }


#else
  
  RWLockC::RWLockC(bool preferWriters)
    : AccM(), 
      RdCount(0), 
      WrWait(0), 
      RdWait(0), 
      WriteQueue(0),
      ReadQueue(0),
      m_preferWriter(preferWriters)
  {} 
  
  RWLockC::RWLockC(const RWLockC &other)
    : AccM(), 
      RdCount(0), 
      WrWait(0), 
      RdWait(0), 
      WriteQueue(0),
      ReadQueue(0),
      m_preferWriter(other.m_preferWriter)
  {} 
  
  bool RWLockC::RdLock()
  {
    AccM.Lock();
    while((WrWait > 0 && m_preferWriter) || RdCount < 0) {
      RdWait++; // Should only go aroung this loop once !
      AccM.Unlock();
      ReadQueue.Wait();
      AccM.Lock();
      RdWait--;
    }
    RavlAssert(RdCount >= 0);
    RdCount++;
    AccM.Unlock();
    return true;
  }

  // Get a read lock.

  bool RWLockC::RdLock(float timeout) {
    DateC timeOutAt = DateC::NowUTC();
    timeOutAt += timeout;
    AccM.Lock();
    while((WrWait > 0 && m_preferWriter) || RdCount < 0) {
      RdWait++; // Should only go aroung this loop once !
      AccM.Unlock();
      float timeToWait = (DateC::NowUTC() - timeOutAt).Double();
      if(timeToWait < 0)
        timeToWait = 0;
      if(!ReadQueue.Wait(timeToWait)) {
        AccM.Lock();
        RdWait--;
        AccM.Unlock();
        return false;
      }
      AccM.Lock();
      RdWait--;
    }
    RavlAssert(RdCount >= 0);
    RdCount++;
    AccM.Unlock();
    return true;
  }


  bool RWLockC::WrLock(void)
  {
    AccM.Lock();
    while(RdCount != 0) {
      WrWait++; // Should only go through here once !
      AccM.Unlock();
      WriteQueue.Wait();
      AccM.Lock();
      WrWait--;
    }
    RdCount = -1; // Flag write lock.
    AccM.Unlock();
    return true;
  }

  //: Aquire a write lock with timeout
  // Returns true if lock aquired, false if timeout.
  // Negative timeout's will cause the wait to last forever

  bool RWLockC::WrLock(float timeout)
  {
    DateC timeOutAt = DateC::NowUTC();
    timeOutAt += timeout;
    AccM.Lock();
    while(RdCount != 0) {
      WrWait++; // Should only go through here once !
      AccM.Unlock();
      float timeToWait = (DateC::NowUTC() - timeOutAt).Double();
      if(timeToWait < 0)
        timeToWait = 0;
      if(!WriteQueue.Wait(timeToWait)) {
        AccM.Lock();
        WrWait--;
        AccM.Unlock();
        return false;
      }
      AccM.Lock();
      WrWait--;
    }
    RdCount = -1; // Flag write lock.
    AccM.Unlock();
    return true;
  }


  bool RWLockC::Unlock(void)
  {
    AccM.Lock();
    if(RdCount < 0) {
      // Unlock a write lock.
      RdCount = 0;
      if(m_preferWriter) {
        if(WrWait > 0) {
    	  WriteQueue.Post(); // Wake up a waiting writer.
        } else {
	  for(int i = 0;i < RdWait;i++)
	    ReadQueue.Post(); // Wakeup all waiting readers.
        }
      } else {
        if(RdWait == 0 && WrWait > 0)
    	  WriteQueue.Post(); // Wake up a waiting writer.
        else {
	  for(int i = 0;i < RdWait;i++)
	    ReadQueue.Post(); // Wakeup all waiting readers.
        }
      }
      AccM.Unlock();
      return true;
    }
    // Unlock a read lock.
    RdCount--;
    if(m_preferWriter) {
      if(WrWait < 1) {
        // No writers waiting so make sure readers are awake
        for(int i = 0;i < RdWait;i++)
	  ReadQueue.Post(); // Wakeup all waiting readers.
      } else {
        // If no readers locking, start a writer.
        if(RdCount <= 0)
	  WriteQueue.Post(); // Wake up a waiting writer.
      }
    } else {
      // Reader preference.
      if(RdWait > 0) {
        // They shouldn't be waiting, but in case.
        for(int i = 0;i < RdWait;i++)
	  ReadQueue.Post(); // Wakeup all waiting readers.
      } else {
        // Nothing waiting, and nothing hold a lock so let
        // writers have a go.
        if(RdCount <= 0)
	  WriteQueue.Post(); // Wake up a waiting writer.
      }
    }
    AccM.Unlock();
    return true;
  }

  bool RWLockC::TryRdLock()  {
    AccM.Lock();
    if(WrWait > 0 || RdCount < 0) {
      AccM.Unlock();
      return false;
    }
    RdCount++;
    AccM.Unlock();
    return true;
  }


  bool RWLockC::TryWrLock(void)  {
    AccM.Lock();
    if(RdCount > 0) {
      AccM.Unlock();
      return false;
    }
    RdCount = -1; // Flag write lock.
    AccM.Unlock();
    return true;
  }


#endif  
  
  //: Print an error.
  
  void RWLockC::Error(const char *msg,int ret) {
    cerr << msg << " (errno=" << errno << ") Return=" << ret << " \n";
    RavlAssert(0); // Abort so we can get a stack trace.
  }

  ostream &operator<<(ostream &strm,const RWLockC &vertex) {
    RavlAssertMsg(0,"not implemented");
    return strm;
  }
  //: Text stream output.
  // Not implemented
  
  istream &operator>>(istream &strm,RWLockC &vertex) {
    RavlAssertMsg(0,"not implemented");
    return strm;
  }
  //: Text stream input.
  // Not implemented
  
  BinOStreamC &operator<<(BinOStreamC &strm,const RWLockC &vertex) {
    RavlAssertMsg(0,"not implemented");
    return strm;
  }
  //: Binary stream output.
  // Not implemented
  
  BinIStreamC &operator>>(BinIStreamC &strm,RWLockC &vertex) {
    RavlAssertMsg(0,"not implemented");
    return strm;
  }
  //: Binary stream input.
  // Not implemented
  
}
