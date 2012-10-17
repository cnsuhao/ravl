// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2011, Charles Galambos
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_ZMQN_SOCKETDISPATCHER_HEADER
#define RAVL_ZMQN_SOCKETDISPATCHER_HEADER 1
//! lib=RavlZmq

#include "Ravl/Zmq/Socket.hh"
#include <zmq.h>

namespace RavlN {
  namespace ZmqN {

    //! Handle events from a socket.

    class SocketDispatcherC
     : public RCBodyVC
    {
    public:
      //! Construct from a socket.
      SocketDispatcherC(const SocketC &socket,bool readReady,bool writeReady);

      //! Factory constructor
      SocketDispatcherC(const XMLFactoryContextC &factory);

      //! Handle event.
      virtual void Dispatch();

      //! Stop handling of events.
      virtual void Stop();

      //! Setup poll item,
      // Return false if socket should be ignored.
      virtual bool SetupPoll(zmq_pollitem_t &pollItem);

      //! Pointer to class
      typedef SmartPtrC<SocketDispatcherC> RefT;

    protected:
      SocketC::RefT m_socket;
      bool m_onReadReady;
      bool m_onWriteReady;
    };


  }
}

#endif
