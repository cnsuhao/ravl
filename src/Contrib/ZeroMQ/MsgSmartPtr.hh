// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2011, Charles Galambos
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! lib=RavlZmq

#ifndef RAVL_ZMQ_MSGSMARTPTR_HH_
#define RAVL_ZMQ_MSGSMARTPTR_HH_

#include "Ravl/SArray1d.hh"
#include "Ravl/Zmq/MsgBuffer.hh"
#include <zmq.h>

namespace RavlN {
  namespace ZmqN {

    //! class for passing smart pointers through zmq
    //! NOTE: This will only work with inproc sockets and classes derived from RavlN::RCBodyC, use with
    // other sockets types cause undefined behaviour.

    template<typename ClassT>
    class MsgSmartPtrC
    {
    public:
      //! Access contained class.
      static ClassT *MsgValue(SArray1dC<char> &msg)
      {
        MsgBufferBodyC *msgBuff = dynamic_cast<MsgBufferBodyC *>(msg.Buffer().BodyPtr());
        return reinterpret_cast<ClassT *>(zmq_msg_data(msgBuff->Msg()));
      }

      //! Access contained class.
      static ClassT *MsgValue(zmq_msg_t &msg)
      {
        return reinterpret_cast<ClassT *>(zmq_msg_data(&msg));
      }

      //! Turn a class pointer into something that looks like an array.
      static SArray1dC<char> AsArray(const ClassT *aClass)
      {
        if(aClass != 0) aClass->IncRefCounter();
        return SArray1dC<char>(MsgBufferC((void *) aClass,sizeof(ClassT),&ZmqFreeFn,0),(char *)aClass,0);
      }

    protected:

      static void ZmqFreeFn(void *data,void *hint) {
        ClassT *theClass = reinterpret_cast<ClassT *>(data);
        if(theClass != 0) {
          if(!theClass->DecRefCounter())
            delete theClass;
        }
      }

    };

  }
}

#endif /* MSGBUFFER_HH_ */
