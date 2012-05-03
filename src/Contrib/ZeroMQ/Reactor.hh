// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2011, Charles Galambos
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_ZMQN_REACTOR_HEADER
#define RAVL_ZMQN_REACTOR_HEADER
//! lib=RavlZmq

#include "Ravl/Zmq/Socket.hh"
#include "Ravl/Zmq/SocketDispatcher.hh"
#include "Ravl/SArray1d.hh"
#include "Ravl/String.hh"
#include "Ravl/Trigger.hh"
#include "Ravl/ServiceThread.hh"
#include <vector>
#include <string>

namespace RavlN {
  namespace ZmqN {

    //! Reactor.

    class ReactorC
      : public ServiceThreadC
    {
    public:
      //! Default constructor.
      ReactorC();

      //! Factory constructor
      ReactorC(const XMLFactoryContextC &factory);

      //! Write to an ostream
      bool Save(std::ostream &strm) const;

      //! Write to a binary stream
      // Not implemented
      bool Save(BinOStreamC &strm) const;

      //! Sugar to make it easier to setup from a factory.
      bool CallOnRead(const XMLFactoryContextC &factory,const std::string &name,const TriggerC &trigger,SocketC::RefT &skt);

      //! Sugar to make it easier to setup from a factory.
      bool CallOnWrite(const XMLFactoryContextC &factory,const std::string &name,const TriggerC &trigger,SocketC::RefT &skt);

      //! Add a read trigger
      bool CallOnRead(const SocketC &socket,const TriggerC &trigger);

      //! Add a write trigger
      bool CallOnWrite(const SocketC &socket,const TriggerC &trigger);

      //! Add handler to system
      bool Add(const SocketDispatcherC &handler);

      //! Remove handler from system
      bool Remove(const SocketDispatcherC &handler);

      //! Run reactor loop.
      virtual bool Run();


      //! Owner reference counted ptr to class
      typedef RavlN::SmartOwnerPtrC<ReactorC> RefT;

      //! Callback reference counter ptr to class
      typedef RavlN::SmartCallbackPtrC<ReactorC> CBRefT;

    protected:
      //! Called by the main loop when its first run.
      virtual bool OnStart();

      //! Called when loop is exiting.
      virtual bool OnFinish();

      std::vector<SocketDispatcherC::RefT> m_sockets;
      float m_teminateCheckInterval;
      bool m_pollListChanged;
      bool m_verbose;
      //! Called when owner handles drop to zero.
      virtual void ZeroOwners();

    };
  }
}

#endif
