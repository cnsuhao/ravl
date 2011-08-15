/*
 * ThreadLimitCounter.cc
 *
 *  Created on: 9 Aug 2011
 *      Author: charles
 */

#include "Ravl/Threads/ThreadLimitCounter.hh"

namespace RavlN {

  //: Constructor
  ThreadLimitCounterC::ThreadLimitCounterC(UIntT count)
   : m_count(count),
     m_maxCount(count),
     m_waiting(0)
  {
    RavlAssert(m_maxCount > 0);
    if(m_maxCount <= 0)
      throw RavlN::ExceptionOutOfRangeC("Count to large.");
  }

  //: Release a number of units back to the pool.
  void ThreadLimitCounterC::Release(UIntT units)
  {
    m_cond.Lock();
    m_count += units;
    RavlAssert(m_count <= m_maxCount);
    m_cond.Unlock();
    m_cond.Broadcast();
  }

  bool ThreadLimitCounterC::Request(UIntT units)
  {
    RavlAssert(((IntT) units) >= 0); // Alert us if we're being passed silly numbers
    if (units > (UIntT) m_maxCount)
      return false;
    m_cond.Lock();
    m_waiting++;
    while (m_count < (IntT) units)
      m_cond.Wait();
    m_count -= units;
    RavlAssert(m_count >= 0);
    UIntT waiting = --m_waiting;
    m_cond.Unlock();
    if(waiting == 0)
      m_cond.Broadcast();
    return true;
  }

  bool ThreadLimitCounterC::RequestWait(UIntT units, float maxTime)
  {
    bool ret = true;
    RavlAssert(((IntT) units) >= 0);// Alert us if we're being passed silly numbers
    m_cond.Lock();
    m_waiting++;
    DateC deadline = DateC::NowUTC() + maxTime;
    while (m_count < (IntT) units)
      ret = m_cond.WaitUntil(deadline);
    if (ret) {
      m_count -= units;
      RavlAssert(m_count >= 0);
    }
    UIntT waiting = --m_waiting;
    m_cond.Unlock();
    if(waiting == 0)
      m_cond.Broadcast();
    return ret;
  }

  //: Wait for lock to be free of all waiters.
  bool ThreadLimitCounterC::WaitForFree() const
  {
    if (m_waiting == 0)
      return true;
    m_cond.Lock();
    while (m_waiting != 0)
      m_cond.Wait();
    m_cond.Unlock();
    return true;
  }

}
