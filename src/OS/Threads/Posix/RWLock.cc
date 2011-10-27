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
  
  RWLockC::RWLockC(RWLockKindT kind)
    : isValid(false),
      m_preferWriter(kind == RWLOCK_PREFER_WRITERS)
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
  
  RWLockC::RWLockC(RWLockKindT kind)
    : m_accM(), 
      m_rdCount(0), 
      m_wrWait(0), 
      m_rdWait(0), 
      m_preferWriter(kind == RWLOCK_PREFER_WRITERS)
  {} 
  
  RWLockC::RWLockC(const RWLockC &other)
    : m_accM(), 
      m_rdCount(0), 
      m_wrWait(0), 
      m_rdWait(0), 
      m_preferWriter(other.m_preferWriter)
  {} 
  
  bool RWLockC::RdLock()
  {
    MutexLockC alock(m_accM);
    while((m_wrWait > 0 && m_preferWriter) || m_rdCount < 0) {
      m_rdWait++; // Should only go around this loop once !
      alock.Unlock();
      m_readQueue.Lock();
      m_readQueue.Wait();
      m_readQueue.Unlock();
      alock.Lock();
      m_rdWait--;
    }
    RavlAssert(m_rdCount >= 0);
    m_rdCount++;
    return true;
  }

  // Get a read lock.

  bool RWLockC::RdLock(float timeout) {
    DateC timeOutAt = DateC::NowUTC();
    timeOutAt += timeout;
    MutexLockC alock(m_accM);
    while((m_wrWait > 0 && m_preferWriter) || m_rdCount < 0) {
      m_rdWait++; // Should only go around this loop once !
      alock.Unlock();
      float timeToWait = static_cast<float>((DateC::NowUTC() - timeOutAt).Double());
      if(timeToWait < 0)
        timeToWait = 0;
      m_readQueue.Lock();
      if(!m_readQueue.Wait(timeToWait)) {
        m_readQueue.Unlock();
        alock.Lock();
        m_rdWait--;
        return false;
      }
      m_readQueue.Unlock();
      alock.Lock();
      m_rdWait--;
    }
    RavlAssert(m_rdCount >= 0);
    m_rdCount++;
    return true;
  }


  bool RWLockC::WrLock(void)
  {
    MutexLockC alock(m_accM);
    while(m_rdCount != 0) {
      m_wrWait++; // Should only go through here once !
      alock.Unlock();
      m_writeQueue.Lock();
      m_writeQueue.Wait();
      m_writeQueue.Unlock();
      alock.Lock();
      m_wrWait--;
    }
    m_rdCount = -1; // Flag write lock.
    return true;
  }

  //: Acquire a write lock with timeout
  // Returns true if lock acquired, false if timeout.
  // Negative timeout's will cause the wait to last forever

  bool RWLockC::WrLock(float timeout)
  {
    DateC timeOutAt = DateC::NowUTC();
    timeOutAt += timeout;
    MutexLockC alock(m_accM);
    while(m_rdCount != 0) {
      m_wrWait++; // Should only go through here once !
      alock.Unlock();
      float timeToWait = static_cast<float>((DateC::NowUTC() - timeOutAt).Double());
      if(timeToWait < 0)
        timeToWait = 0;
      m_writeQueue.Lock();
      if(!m_writeQueue.Wait(timeToWait)) {
        m_writeQueue.Unlock();
        alock.Lock();
        m_wrWait--;
        return false;
      }
      m_writeQueue.Unlock();
      alock.Lock();
      m_wrWait--;
    }
    m_rdCount = -1; // Flag write lock.
    return true;
  }


  bool RWLockC::Unlock(void)
  {
    MutexLockC alock(m_accM);
    if(m_rdCount < 0) {
      // Unlock a write lock.
      m_rdCount = 0;
      if(m_preferWriter) {
        if(m_wrWait > 0) {
    	  m_writeQueue.Signal(); // Wake up a waiting writer.
        } else {
	  m_readQueue.Broadcast(); // Wake up all waiting readers.
        }
      } else {
        if(m_rdWait == 0 && m_wrWait > 0)
    	  m_writeQueue.Signal(); // Wake up a waiting writer.
        else {
          m_readQueue.Broadcast(); // Wake up all waiting readers.
        }
      }
      return true;
    }
    // Unlock a read lock.
    m_rdCount--;
    if(m_preferWriter) {
      if(m_wrWait < 1) {
        // No writers waiting so make sure readers are awake
        m_readQueue.Broadcast(); // Wake up all waiting readers.
      } else {
        // If no readers locking, start a writer.
        if(m_rdCount <= 0)
          m_writeQueue.Signal(); // Wake up a waiting writer.
      }
    } else {
      // Reader preference.
      if(m_rdWait > 0) {
        // They shouldn't be waiting, but in case.
	m_readQueue.Broadcast(); // Wake up all waiting readers.
      } else {
        // Nothing waiting, and nothing holding a lock so let
        // writers have a go.
        if(m_rdCount <= 0)
	  m_writeQueue.Signal(); // Wake up a waiting writer.
      }
    }
    return true;
  }

  bool RWLockC::TryRdLock()  {
    MutexLockC alock(m_accM);
    if(m_wrWait > 0 || m_rdCount < 0)
      return false;
    m_rdCount++;
    return true;
  }


  bool RWLockC::TryWrLock(void)  {
    MutexLockC alock(m_accM);
    if(m_rdCount > 0)
      return false;
    m_rdCount = -1; // Flag write lock.
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
