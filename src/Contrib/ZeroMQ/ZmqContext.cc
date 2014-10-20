// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2011, Charles Galambos
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! lib=RavlZmq

#include "Ravl/Zmq/Context.hh"
#include "Ravl/XMLFactoryRegister.hh"
#include <zmq.h>

namespace RavlN { namespace ZmqN {

    //! Constructor.
    ContextC::ContextC(int numThreads)
    {
      m_zmqContext = zmq_init (numThreads);
    }

      //! Constructor.
    ContextC::ContextC(const XMLFactoryContextC &context)
    {
      int numThreads = context.AttributeInt("threads",1);
      m_zmqContext = zmq_init (numThreads);

      int numSockets = context.AttributeInt("maxSockets",-1);
      if(numSockets >= 0 ) {
        zmq_ctx_set (m_zmqContext, ZMQ_MAX_SOCKETS,numSockets);
      }
    }

    //! Destructor.
    ContextC::~ContextC()
    {
      if(m_zmqContext != 0)
        zmq_term(m_zmqContext);
      m_zmqContext = 0;
    }

    //! Write to an ostream
    bool ContextC::Save(std::ostream &strm) const {
      return true;
    }

    //! Write to a binary stream
    // Not implemented
    bool ContextC::Save(BinOStreamC &strm) const {
      RavlAssertMsg(0,"not supported");
      return false;
    }

    void LinkContext()
    {}

    XMLFactoryRegisterC<ContextC> g_registerZmqContext("RavlN::ZmqN::ContextC");
}}

