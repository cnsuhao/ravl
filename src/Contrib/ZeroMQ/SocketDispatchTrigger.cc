// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2011, Charles Galambos
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! lib=RavlZmq

#include "Ravl/Zmq/SocketDispatchTrigger.hh"
#include "Ravl/XMLFactoryRegister.hh"

namespace RavlN {
  namespace ZmqN {

    //! Construct from a socket.
    SocketDispatchTriggerC::SocketDispatchTriggerC(const SocketC &socket,bool readReady,bool writeReady,const TriggerC &trigger)
     : SocketDispatcherC(socket,readReady,writeReady),
       m_trigger(trigger)
    {
    }

    //! Factory constructor
    SocketDispatchTriggerC::SocketDispatchTriggerC(const XMLFactoryContextC &factory)
     : SocketDispatcherC(factory)
    {
    }

    //! Handle event.
    void SocketDispatchTriggerC::Dispatch()
    {
      RavlAssert(m_trigger.IsValid());
      m_trigger();
    }

  }
}
