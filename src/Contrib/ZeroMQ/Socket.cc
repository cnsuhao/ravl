// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2011, Charles Galambos
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! lib=RavlZmq

#include "Ravl/Zmq/Socket.hh"
#include "Ravl/Zmq/Context.hh"
#include "Ravl/Zmq/MsgBuffer.hh"
#include "Ravl/OS/SysLog.hh"
#include "Ravl/Exception.hh"
#include "Ravl/XMLFactoryRegister.hh"

namespace RavlN {
  namespace ZmqN {

    static const char *g_socketTypeNames[] =
    {
      "PAIR",
      "PUB",
      "SUB",
      "REQ",
      "REP",
      "XREQ",
      "XREP",
      "PULL",
      "PUSH",
      "XPUB",
      "XSUB",
      "ROUTER",
      "DEALER",
      "Unknown1",
      "Unknown2",
      "Unknown3",
      "Unknown4",
      0
    };

    //! Write to text stream.
    std::ostream &operator<<(std::ostream &strm,SocketTypeT sockType)
    {
      if((int) sockType >= 15) {
        strm << "UnknownX";
        return strm;
      }
      strm << g_socketTypeNames[(int) sockType];
      return strm;
    }

    //! Convert a socket type from a name to an enum
    SocketTypeT SocketType(const std::string &name) {
      int i = 0;
      for(;g_socketTypeNames[i] != 0;i++) {
        if(name == g_socketTypeNames[i]) {
          return static_cast<SocketTypeT>(i);
        }
      }
      RavlError("Unknown socket type '%s' ",name.data());
      throw ExceptionC("Unknown socket type");
    }

    //! Read from text stream
    std::istream &operator>>(std::istream &strm,SocketTypeT &sockType)
    {
      std::string socketTypeName;
      strm >> socketTypeName;
      sockType = SocketType(socketTypeName);
      return strm;
    }


    //! Construct a new socket.
    SocketC::SocketC(ContextC &context,SocketTypeT socketType,const StringC &codec)
     : m_socket(0),
       m_defaultCodec(codec),
       m_verbose(false)
    {
      m_socket = zmq_socket(context.RawContext(),(int) socketType);
      if(m_socket == 0) {
        RavlError("Failed to create socket: %s ",zmq_strerror (zmq_errno ()));
        throw ExceptionOperationFailedC("Failed to create socket. ");
      }
    }


    //! Factory constructor
    SocketC::SocketC(const XMLFactoryContextC &context)
     : m_name(context.Path().data()),
       m_socket(0),
       m_defaultCodec(context.AttributeString("defaultCodec","")),
       m_verbose(context.AttributeBool("verbose",false))
    {
      ContextC::RefT ctxt;
      if(!context.UseComponent("ZmqContext",ctxt,false,typeid(ZmqN::ContextC))) {
        RavlError("No context for socket at %s ",context.Path().data());
        throw ExceptionOperationFailedC("No context. ");
      }
      SocketTypeT sockType = SocketType(context.AttributeString("socketType",""));
      m_socket = zmq_socket(ctxt->RawContext(),(int) sockType);
      if(m_socket == 0) {
        RavlError("Failed to create socket '%s' in %s ",zmq_strerror (zmq_errno ()),context.Path().data());
        throw ExceptionOperationFailedC("Failed to create socket. ");
      }

      // Bind if required.
      StringC defaultBind = context.AttributeString("bind","");
      if(!defaultBind.IsEmpty()) {
        Bind(defaultBind);
      }

      // Connnect if required.
      StringC defaultConnect = context.AttributeString("connect","");
      if(!defaultConnect.IsEmpty()) {
        Connect(defaultConnect);
      }

      // Setup properties
      {
        XMLFactoryContextC childContext;
        if(context.ChildContext("Properties",childContext))
        for(RavlN::DLIterC<XMLTreeC> it(childContext.Children());it;it++) {
          if(it->Name() == "Subscribe") {
            Subscribe(it->AttributeString("value","").data());
            continue;
          }
          if(it->Name() == "Bind") {
            Bind(it->AttributeString("value","").data());
            continue;
          }
          if(it->Name() == "Connect") {
            Connect(it->AttributeString("value","").data());
            continue;
          }
          if(it->Name() == "Identity") {
            SetIdentity(it->AttributeString("value",""));
            continue;
          }
          if(it->Name() == "Linger") {
            SetLinger(it->AttributeReal("value",-1.0));
            continue;
          }
          if(it->Name() == "HighWaterMark") {
            SetHighWaterMark(it->AttributeUInt("value",0));
            continue;
          }
          RavlError("Unknown socket property '%s' at %s ",it->Name().data(),childContext.Path().data());
          throw ExceptionOperationFailedC("Unknown property. ");
        }
      }
    }

    //! Destructor
    SocketC::~SocketC()
    {
      if(m_socket != 0)
        zmq_close (m_socket);
    }

    //! Write to an ostream
    bool SocketC::Save(std::ostream &strm) const
    {
      RavlAssertMsg(0,"not implemented");
      return false;
    }

    //! Write to a binary stream
    // Not implemented
    bool SocketC::Save(BinOStreamC &strm) const
    {
      RavlAssertMsg(0,"not supported");
      return false;
    }

    //! Bind to an address
    void SocketC::Bind(const std::string &addr)
    {
      RavlAssert(m_socket != 0);
      int ret;
      if((ret = zmq_bind (m_socket, addr.data())) != 0) {
        RavlError("Failed to bind to %s : %s ",addr.data(),zmq_strerror (zmq_errno ()));
        throw ExceptionOperationFailedC("bind failed. ");
      }
    }

    //! Connect to an address.
    void SocketC::Connect(const std::string &addr)
    {
      RavlAssert(m_socket != 0);
      int ret;
      if((ret = zmq_connect(m_socket, addr.data())) != 0) {
        RavlError("Failed to connect to %s : %s ",addr.data(),zmq_strerror (zmq_errno ()));
        throw ExceptionOperationFailedC("connect failed. ");
      }
    }

    //! Set linger time for socket.
    void SocketC::SetLinger(float timeSeconds)
    {
      RavlAssert(m_socket != 0);
      int lingerTime = 0;
      if(timeSeconds >= 0) {
        lingerTime = Round(timeSeconds) * 1000.0;
      } else {
        lingerTime = -1;
      }
      int ret;
      if((ret = zmq_setsockopt (m_socket,ZMQ_LINGER,&lingerTime,sizeof(lingerTime))) != 0) {
        RavlError("Failed to set linger to %f : %s ",timeSeconds,zmq_strerror (zmq_errno ()));
        throw ExceptionOperationFailedC("connect failed. ");
      }
    }

    //! Set the high water mark.
    void SocketC::SetHighWaterMark(UInt64T queueLimit) {
      RavlAssert(m_socket != 0);
      int ret;
#ifdef ZMQ_HWM
      if((ret = zmq_setsockopt (m_socket,ZMQ_HWM,&queueLimit,sizeof(queueLimit))) != 0) {
        RavlError("Failed to set high water mark to %u : %s ",(UIntT) queueLimit,zmq_strerror (zmq_errno ()));
        throw ExceptionOperationFailedC("connect failed. ");
      }
#else
      if((ret = zmq_setsockopt (m_socket,ZMQ_SNDHWM,&queueLimit,sizeof(queueLimit))) != 0) {
        RavlError("Failed to set high water mark to %u : %s ",(UIntT) queueLimit,zmq_strerror (zmq_errno ()));
        throw ExceptionOperationFailedC("connect failed. ");
      }
      if((ret = zmq_setsockopt (m_socket,ZMQ_RCVHWM,&queueLimit,sizeof(queueLimit))) != 0) {
        RavlError("Failed to set high water mark to %u : %s ",(UIntT) queueLimit,zmq_strerror (zmq_errno ()));
        throw ExceptionOperationFailedC("connect failed. ");
      }
#endif
    }

    //! Subscribe to a topic
    void SocketC::Subscribe(const std::string &topic)
    {
      RavlAssert(m_socket != 0);
      RavlDebug("Subscribing '%s' to '%s' ",m_name.data(),topic.data());
      int ret = zmq_setsockopt (m_socket,ZMQ_SUBSCRIBE,topic.data(),topic.size());
      if(ret != 0) {
        RavlError("Failed to subscribe to %s : %s ",topic.data(),zmq_strerror (zmq_errno ()));
        throw ExceptionOperationFailedC("Subscribe failed. ");
      }
    }

    //! Unsubscribe from a topic
    bool SocketC::Unsubscribe(const std::string &topic)
    {
      RavlAssert(m_socket != 0);
      RavlDebug("Unsubscribing '%s' from %s ",m_name.data(),topic.data());
      int ret = zmq_setsockopt (m_socket,ZMQ_UNSUBSCRIBE,topic.data(),topic.size());
      if(ret != 0) {
        RavlError("Failed to unsubscribe to %s : %s ",topic.data(),zmq_strerror (zmq_errno ()));
        throw ExceptionOperationFailedC("Unsubscribe failed. ");
      }
      return true;
    }

    //! Set the identity of the socket.
    void SocketC::SetIdentity(const std::string &identity) {
      RavlAssert(m_socket != 0);
      RavlDebug("Setting identity to %s ",identity.data());
      int ret = zmq_setsockopt (m_socket,ZMQ_IDENTITY,identity.data(),identity.size());
      if(ret != 0) {
        RavlError("Failed to set identity to %s : %s ",identity.data(),zmq_strerror (zmq_errno ()));
        throw ExceptionOperationFailedC("Unsubscribe failed. ");
      }
    }

    //! Send a message
    bool SocketC::Send(const SArray1dC<char> &msg,BlockT block) {
      zmq_msg_t zmsg;
      if(m_verbose) {
        StringC tmp(msg.ReferenceElm(),msg.Size(),msg.Size());
        RavlDebug("Send %s:'%s'",m_name.data(),tmp.data());
      }
      ArrayToMessage(zmsg,msg);
      int ret;
      int flags = 0;
#if ZMQ_VERSION_MAJOR > 2
      if((ret = zmq_sendmsg (m_socket, &zmsg, flags)) != 0)
#else
      if((ret = zmq_send (m_socket, &zmsg, flags)) != 0)
#endif
      {
        int anErrno = zmq_errno();
        if(block == ZSB_NOBLOCK && (anErrno == EAGAIN || anErrno == EINTR))
          return false;
        RavlError("Send failed : %s   flags=%x errno=%d ",zmq_strerror (anErrno),flags,anErrno);
        zmq_msg_close(&zmsg);
#if 0
        throw ExceptionOperationFailedC("Send failed. ");
#else
        return false;
#endif
      }
      zmq_msg_close(&zmsg);
      return true;
    }

    //! Receive a message.
    bool SocketC::Recieve(SArray1dC<char> &msg,BlockT block)
    {
      int64_t more = 0;
      int ret;
      int flags = 0;
      static size_t more_size = sizeof (more);
      MsgBufferC msgBuffer(0);

#if ZMQ_VERSION_MAJOR >= 3
      if((ret = zmq_recvmsg (m_socket, msgBuffer.Msg(), flags)) != 0)
#else
      if((ret = zmq_recv (m_socket, msgBuffer.Msg(), flags)) != 0)
#endif
      {
        int anErrno = zmq_errno();
        RavlError("Recv failed : %s ",zmq_strerror (anErrno));
        throw ExceptionOperationFailedC("Recv failed. ");
      }
      msgBuffer.Build();
      msg = SArray1dC<char>(msgBuffer,msgBuffer.Size());

      if((ret = zmq_getsockopt (m_socket, ZMQ_RCVMORE, &more, &more_size)) != 0) {
        int anErrno = zmq_errno();
        RavlError("RCVMORE failed : %s ",zmq_strerror (anErrno));
        throw ExceptionOperationFailedC("Recv failed. ");
      }
      while(more)  {
        zmq_msg_t zmsg;
        zmq_msg_init(&zmsg);
        RavlWarning("Discarding message part.");
#if ZMQ_VERSION_MAJOR >= 3
        if((ret = zmq_recvmsg (m_socket, msgBuffer.Msg(), flags)) != 0)
#else
        if((ret = zmq_recv (m_socket, msgBuffer.Msg(), flags)) != 0)
#endif
        {
          int anErrno = zmq_errno();
          RavlError("RCVMORE failed : %s ",zmq_strerror (anErrno));
          throw ExceptionOperationFailedC("Recv failed. ");
        }
        zmq_msg_close(&zmsg);

        if((ret = zmq_getsockopt (m_socket, ZMQ_RCVMORE, &more, &more_size)) != 0) {
          int anErrno = zmq_errno();
          RavlError("RCVMORE failed : %s ",zmq_strerror (anErrno));
          throw ExceptionOperationFailedC("Recv failed. ");
        }
      }
      if(m_verbose) {
        StringC tmp(msg.ReferenceElm(),msg.Size(),msg.Size());
        RavlDebug("Recieved %s:'%s'",m_name.data(),tmp.data());
      }
      return true;
    }

    //! Send a message
    bool SocketC::Send(const MessageC &msg,BlockT block)
    {
      RavlAssert(m_socket != 0);
      size_t elems = msg.Parts().size();
      RavlAssert(elems > 0);
      size_t lastElem = elems -1;
      int ret;
      //RavlDebug("Sending %zu parts.",elems);
      for(size_t i = 0;i < elems;i++) {
        zmq_msg_t zmsg;
        ArrayToMessage(zmsg,msg.Parts()[i]);

#ifdef ZMQ_DONTWAIT
       int flags = block == ZSB_BLOCK ? 0 : ZMQ_DONTWAIT;
#else
       int flags = block == ZSB_BLOCK ? 0 : ZMQ_NOBLOCK;
#endif

        if(i < lastElem) {
          flags |= ZMQ_SNDMORE;
        }
#if ZMQ_VERSION_MAJOR > 2
        if((ret = zmq_sendmsg (m_socket, &zmsg, flags)) != 0)
#else
        if((ret = zmq_send (m_socket, &zmsg, flags)) != 0)
#endif
        {
          int anErrno = zmq_errno();
          if(i == 0 && block == ZSB_NOBLOCK && (anErrno == EAGAIN || anErrno == EINTR))
            return false;
          RavlError("Send failed : %s   flags=%x errno=%d ",zmq_strerror (anErrno),flags,anErrno);
          zmq_msg_close(&zmsg);
#if 0
          throw ExceptionOperationFailedC("Send failed. ");
#else
          return false;
#endif
        }
        zmq_msg_close(&zmsg);
      }
      return true;
    }


    //! Receive a message.
    bool SocketC::Recieve(MessageC::RefT &msg,BlockT block)
    {
      RavlAssert(m_socket != 0);
      msg = new MessageC();
      int64_t more = 0;
      int ret;
      static size_t more_size = sizeof (more);
      do {
        MsgBufferC msgBuffer(0);
        int flags = 0;

#if ZMQ_VERSION_MAJOR >= 3
        if((ret = zmq_recvmsg (m_socket, msgBuffer.Msg(), flags)) != 0)
#else
        if((ret = zmq_recv (m_socket, msgBuffer.Msg(), flags)) != 0)
#endif
        {
          int anErrno = zmq_errno();
          RavlError("Recv failed : %s ",zmq_strerror (anErrno));
          throw ExceptionOperationFailedC("Recv failed. ");
        }
        msgBuffer.Build();
        msg->Parts().push_back(SArray1dC<char>(msgBuffer,msgBuffer.Size()));

        if((ret = zmq_getsockopt (m_socket, ZMQ_RCVMORE, &more, &more_size)) != 0) {
          int anErrno = zmq_errno();
          RavlError("RCVMORE failed : %s ",zmq_strerror (anErrno));
          throw ExceptionOperationFailedC("Recv failed. ");
        }
	//RavlDebug("More: %d Size:%zu ",more,msgBuffer.Size().V());
      } while(more) ;
      return true;
    }

    void LinkSocket()
    {}

    static XMLFactoryRegisterC<SocketC> g_regiserSocket("RavlN::ZmqN::SocketC");

}}
