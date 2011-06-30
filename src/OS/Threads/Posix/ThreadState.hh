// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2011, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_THREADSTATE_HEADER
#define RAVL_THREADSTATE_HEADER 1
/////////////////////////////////////////////////
//! file="Ravl/OS/Threads/Posix/ThreadState.hh"
//! lib=RavlThreads
//! userlevel=Normal
//! docentry="Ravl.API.OS.Threads"
//! author="Charles Galambos"
//! date="25/05/2011"

#include "Ravl/config.h"
#include "Ravl/Threads/ConditionalMutex.hh"
#include "Ravl/Stream.hh"

namespace RavlN
{

  //! userlevel=Devel
  //: Base class for ThreadStateC

  class ThreadStateBaseC
  {
  public:
    ThreadStateBaseC();
    //: Constructor

    ~ThreadStateBaseC();
    //: Destructor.

    bool WaitForFree();
    //: Wait for lock to be free of all waiters.

    IntT ThreadsWaiting() const
    { return m_waiting; }
    //: Get approximation of number of threads waiting.

  protected:
    ConditionalMutexC m_cond;
    volatile IntT m_waiting; // Count of number of threads waiting on this...
  };

  //! userlevel=Normal
  //: Thread safe state machine.
  // <p>This class enables a thread to sleep until a specific state or set of states has been entered.</p>
  // It is assumed that 'StateT' may be copied without a lock, as is the case with built and enumerated types.

  template<typename StateT>
  class ThreadStateC
   : public ThreadStateBaseC
  {
  public:
    ThreadStateC(const StateT &initialState)
      : m_state(initialState)
    {}
    //: Constructor

    bool Update(const StateT &newState) {
      m_cond.Lock();
      if(newState == m_state) {
        // No change.
        m_cond.Unlock();
        return false;
      }
      m_state = newState;
      m_cond.Unlock();
      m_cond.Broadcast();
      return true;
    }
    //: Set the current state.
    // Returns true if new state has changed.  false it its at the given state already

    bool Update(const StateT &expectedCurrentState,const StateT &newState) {
      m_cond.Lock();
      if(newState != expectedCurrentState) {
        // No change.
        m_cond.Unlock();
        return false;
      }
      m_state = newState;
      m_cond.Unlock();
      m_cond.Broadcast();
      return true;
    }
    //: Change from a given existing state to a new state
    // Returns true if event transition was achieved.
    // The state of class isn't 'expectedCurrentState' return false.

    void Wait(const StateT &desiredState) {
      if(m_state == desiredState) // Check before we bother with locking.
        return ;
      m_cond.Lock();
      m_waiting++;
      while(m_state != desiredState)
        m_cond.Wait();
      int value = --m_waiting;
      m_cond.Unlock();
      if(value == 0)
        m_cond.Broadcast(); // If something is waiting for it to be free...
    }
    //: Wait indefinitely for an event to be posted.


    bool Wait(RealT maxTime,const StateT &desiredState)
    {
      if(m_state == desiredState) // Check before we bother with locking.
        return true;
      bool ret = true;
      m_cond.Lock();
      m_waiting++;
      while(m_state != desiredState && ret)
        ret = m_cond.Wait(maxTime);
      int value = --m_waiting;
      m_cond.Unlock();
      if(value == 0)
        m_cond.Broadcast(); // If something is waiting for it to be free...
      return ret;
    }
    //: Wait for a given state.
    // Returns false if timed out.

    bool Wait(RealT maxTime,
               const StateT &desiredState1,
               const StateT &desiredState2,
               StateT &stateAchieved
               )
    {
      if(m_state == desiredState1 || m_state == desiredState2) // Check before we bother with locking.
        return true;
      bool ret = true;
      m_cond.Lock();
      m_waiting++;
      while(m_state != desiredState1 && m_state != desiredState2 && ret)
        ret = m_cond.Wait(maxTime);
      int value = --m_waiting;
      stateAchieved = m_state;
      m_cond.Unlock();
      if(value == 0)
        m_cond.Broadcast(); // If something is waiting for it to be free...
      return ret;
    }
    //: Wait for one of two states to be reached.
    // Returns false if timed out.

    bool Wait(RealT maxTime,
               const StateT &desiredState1,
               const StateT &desiredState2,
               const StateT &desiredState3,
               StateT &stateAchieved
               )
    {
      if(m_state == desiredState1 || m_state == desiredState2 || m_state == desiredState3) // Check before we bother with locking.
        return true;
      bool ret = true;
      m_cond.Lock();
      m_waiting++;
      while(m_state != desiredState1 && m_state != desiredState2 && m_state != desiredState3 && ret)
        ret = m_cond.Wait(maxTime);
      int value = --m_waiting;
      stateAchieved = m_state;
      m_cond.Unlock();
      if(value == 0)
        m_cond.Broadcast(); // If something is waiting for it to be free...
      return ret;
    }
    //: Wait for one of two states to be reached.
    // Returns false if timed out.

    operator StateT () const
    { return m_state; }
    //: Access state

    const StateT State() const
    { return m_state; }
    //: Access the current state.

  protected:
    volatile StateT m_state;
  };
};

#endif
