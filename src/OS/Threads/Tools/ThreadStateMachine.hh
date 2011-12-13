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
#include "Ravl/Threads/Signal1.hh"
#include "Ravl/OS/Date.hh"

namespace RavlN
{

  //! userlevel=Normal
  //: Thread safe state machine.
  // <p>This class enables a thread to sleep until a specific state or set of states has been entered.</p>
  // It is assumed that 'StateT' may be copied without a lock, as is the case with built and enumerated types.

  template<typename StateT,typename TransitionT>
  class ThreadStateMachineC
   : public ThreadStateC<StateT>
  {
  public:
    ThreadStateMachineC(const StateT &initialState,const TransitionT &transitionTable)
     : ThreadStateC<StateT>(initialState),
       m_transitions(transitionTable)
    {}
    //: Constructor

    bool ProcessEvent(const EventT &event) {
      m_cond.Lock();
      StateT newState;
      try {
        newState = m_transitions(m_state,event);
      } catch (...) {
        m_cond.Unlock();
        throw ;
      }
      if(newState == m_state) {
        m_cond.Unlock();
        return false;
      }
      m_state = newState;
      m_cond.Unlock();
      m_cond.Broadcast();
      m_sigState(newState);
      return true;
    }
    //: Process an event

    const std::map<StateT,std::map<EventT,StateT> > &Transitions() const
    { return m_transitions; }
    //: Access transition table.
  protected:
    TransitionT m_transitions;
  };
};

#endif
