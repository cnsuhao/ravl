// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2011, Charles Galambos
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! lib=RavlZmq

#include "Ravl/Zmq/SocketDispatcher.hh"
#include "Ravl/XMLFactoryRegister.hh"

namespace RavlN {
  namespace ZmqN {

    //! Construct from a socket.
    SocketDispatcherC::SocketDispatcherC(const SocketC &socket,bool readReady,bool writeReady)
     : m_socket(&socket),
       m_fd(-1),
       m_onReadReady(readReady),
       m_onWriteReady(writeReady),
       m_onError(false)
    {
    }

    //! Construct from a file descriptor
    SocketDispatcherC::SocketDispatcherC(int fd,bool readReady,bool writeReady,bool onError)
    : m_fd(fd),
      m_onReadReady(readReady),
      m_onWriteReady(writeReady),
      m_onError(onError)
    {

    }

    //! Factory constructor
    SocketDispatcherC::SocketDispatcherC(const XMLFactoryContextC &factory)
     : m_fd(-1),
       m_onReadReady(factory.AttributeBool("onRead",true)),
       m_onWriteReady(factory.AttributeBool("onWrite",false)),
       m_onError(factory.AttributeBool("onError",false))
    {
      rThrowBadConfigContextOnFailS(factory,UseComponent("Socket",m_socket),"No socket given");
    }

    //! Handle event.
    void SocketDispatcherC::Dispatch()
    {
      RavlAssertMsg(0,"Abstract method called.");
    }

    //! Stop handling of events.
    void SocketDispatcherC::Stop() {

    }

    //! Setup poll item,
    // Return false if socket should be ignored.

    bool SocketDispatcherC::SetupPoll(zmq_pollitem_t &pollItem)
    {
      RavlAssert(m_socket.IsValid() || m_fd >= 0);
      if(m_socket.IsValid()) {
        pollItem.socket = m_socket->RawSocket();
        pollItem.fd = -1;
      } else {
        pollItem.socket = 0;
        pollItem.fd = m_fd;
      }
      pollItem.events = 0;
      if(m_onReadReady) {
        pollItem.events |= ZMQ_POLLIN;
      }
      if(m_onWriteReady) {
        pollItem.events |= ZMQ_POLLOUT;
      }
      if(m_onError) {
        pollItem.events |= ZMQ_POLLERR;
      }
      return m_onReadReady || m_onWriteReady || m_onError;
    }


    // -------------------------------------------------------------------

    //! Default constructor
    SocketDispatchSetC::SocketDispatchSetC()
     : RavlN::CollectionC<SocketDispatcherC::RefT>(8)
    {}

    //! Destructors
    SocketDispatchSetC::~SocketDispatchSetC()
    {
      DisconnectAll();
    }

    //! Disconnect all dispatchers.
    void SocketDispatchSetC::DisconnectAll()
    {
      for(int i = 0;i < Size();i++)
        (*this)[i]->Stop();
      Empty();
    }

  }
}
