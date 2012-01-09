// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2011, Charles Galambos
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here

#include "Ravl/Zmq/SocketDispatcher.hh"
#include "Ravl/XMLFactoryRegister.hh"

namespace RavlN {
  namespace ZmqN {

    //! Construct from a socket.
    SocketDispatcherC::SocketDispatcherC(const SocketC &socket,bool readReady,bool writeReady)
     : m_socket(&socket),
       m_onReadReady(readReady),
       m_onWriteReady(writeReady)
    {
    }

    //! Factory constructor
    SocketDispatcherC::SocketDispatcherC(const XMLFactoryContextC &factory)
     : m_onReadReady(factory.AttributeBool("onRead",true)),
       m_onWriteReady(factory.AttributeBool("onWrite",false))
    {
      rThrowBadConfigContextOnFailS(factory,UseComponent("Socket",m_socket),"No socket given");
    }

    //! Handle event.
    void SocketDispatcherC::Dispatch()
    {
      RavlAssertMsg(0,"Abstract method called.");
    }

    //! Setup poll item,
    // Return false if socket should be ignored.

    bool SocketDispatcherC::SetupPoll(zmq_pollitem_t &pollItem)
    {
      RavlAssert(m_socket.IsValid());
      pollItem.socket = m_socket->RawSocket();
      pollItem.events = 0;
      if(m_onReadReady) {
        pollItem.events |= ZMQ_POLLIN;
      }
      if(m_onWriteReady) {
        pollItem.events |= ZMQ_POLLOUT;
      }
      return m_onReadReady || m_onWriteReady;
    }

  }
}
