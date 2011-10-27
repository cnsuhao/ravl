// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
/////////////////////////////////////////////////
//! lib=RavlThreads
//! file="Ravl/OS/Threads/Posix/ThreadEvent.cc"

#include "Ravl/Threads/ThreadEvent.hh"
#include "Ravl/OS/Date.hh"

namespace RavlN
{

  ThreadEventC::ThreadEventC()
    : occurred(false),
      m_waiting(0)
  {}

  bool ThreadEventC::Post() {
    cond.Lock();
    if(occurred) {
      cond.Unlock();
      return false;
    }
    occurred = true;
    cond.Unlock();
    cond.Broadcast();
    return true;
  }
  //: Post an event.
  // Returns true, if event has been posted by this thread.

  ThreadEventC::~ThreadEventC()  {
    if(!occurred)
      Post();
    if(m_waiting != 0)
      cerr << "PThread::~ThreadEvent(), WARNING: Called while threads waiting. \n";
  }
  //: Destructor.

  //: Wait indefinitely for an event to be posted.
  void ThreadEventC::Wait() {
    if(occurred) // Check before we bother with locking.
      return ;
    cond.Lock();
    m_waiting++;
    while(!occurred)
      cond.Wait();
    m_waiting--;
    if(m_waiting == 0) {
      cond.Unlock();
      cond.Broadcast(); // If something is waiting for it to be free...
      return ;
    }
    cond.Unlock();
  }

  //: Wait for an event.
  // Returns false if timed out.
  
  bool ThreadEventC::Wait(RealT maxTime) {
    if(occurred) // Check before we bother with locking.
      return true;
    bool ret = true;
    cond.Lock();
    m_waiting++;
    DateC deadline = DateC::NowUTC() + maxTime;
    while(!occurred && ret) 
      ret = cond.WaitUntil(deadline);
    m_waiting--;
    if(m_waiting == 0) {
      cond.Unlock();
      cond.Broadcast(); // If something is waiting for it to be free...
      return ret;
    }
    cond.Unlock();
    return ret;
  }

  //: Wait for an event.
  // Returns false if timed out.

  bool ThreadEventC::WaitUntil(const DateC &deadline) {
    if(occurred) // Check before we bother with locking.
      return true;
    bool ret(true);
    cond.Lock();
    m_waiting++;
    while(!occurred && ret)
      ret = cond.WaitUntil(deadline);
    m_waiting--;
    if(m_waiting == 0) {
      cond.Unlock();
      cond.Broadcast(); // If something is waiting for it to be free...
      return ret;
    }
    cond.Unlock();
    return ret;
  }

  //: Wait for lock to be free.
  // NB. This is only guaranteed to work for one thread.
  
  bool ThreadEventC::WaitForFree() {
    if(m_waiting == 0)
      return true;
    cond.Lock();
    while(m_waiting != 0) 
      cond.Wait();
    cond.Unlock();
    return true;
  }
  
  
}
