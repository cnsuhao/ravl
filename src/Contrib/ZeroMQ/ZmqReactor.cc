// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2011, Charles Galambos
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! lib=RavlZmq

#include "Ravl/Zmq/Reactor.hh"
#include "Ravl/Zmq/SocketDispatchTrigger.hh"
#include "Ravl/OS/SysLog.hh"
#include "Ravl/XMLFactoryRegister.hh"

namespace RavlN {
  namespace ZmqN {

    //! Discard a message from a socket
    static bool DiscardMessage(ZmqN::SocketC::RefT &skt)
    {
      MessageC::RefT msg;
      skt->Recieve(msg);
      return true;
    }

    //! Default constructor.
    ReactorC::ReactorC(ContextC &context)
     : m_teminateCheckInterval(5.0),
       m_pollListChanged(true),
       m_verbose(false),
       m_timedQueue(false)
    {
      Init(context);
    }


    //! Factory constructor
    ReactorC::ReactorC(const XMLFactoryContextC &factory)
     : ServiceThreadC(factory),
       m_teminateCheckInterval(factory.AttributeReal("terminateCheckInterval",5.0)),
       m_pollListChanged(true),
       m_verbose(factory.AttributeBool("verbose",false)),
       m_timedQueue(false)
    {
      ContextC::RefT context;
      factory.UseComponent("ZmqContext",context);
      Init(*context);
    }

    //! Write to an ostream
    bool ReactorC::Save(std::ostream &strm) const
    {
      return ServiceThreadC::Save(strm);
    }

    //! Write to a binary stream
    // Not implemented
    bool ReactorC::Save(BinOStreamC &strm) const
    {
      RavlAssertMsg(0,"not supported");
      return false;
    }

    //! Do some initial setup
    void ReactorC::Init(ContextC &context)
    {
      m_wakeup = new ZmqN::SocketC(context,ZST_PAIR);
      RavlN::StringC sktId;
      sktId.form("inproc://Reactor-%p",this);
      m_wakeup->Bind(sktId);

      ZmqN::SocketC::RefT wakeupLocal = new ZmqN::SocketC(context,ZST_PAIR);
      wakeupLocal->Connect(sktId);
      CallOnRead(*wakeupLocal,RavlN::Trigger(&DiscardMessage,wakeupLocal));

      m_wakeMsg = new MessageC("");
    }

    //! Sugar to make it easier to setup from a factory.
    SocketDispatcherC::RefT ReactorC::CallOnRead(const XMLFactoryContextC &factory,const std::string &name,const TriggerC &trigger,SocketC::RefT &skt)
    {
      if(!factory.UseComponent(name,skt,false,typeid(ZmqN::SocketC)))
        return 0;
      return CallOnRead(*skt,trigger);
    }

    //! Sugar to make it easier to setup from a factory.
    SocketDispatcherC::RefT ReactorC::CallOnWrite(const XMLFactoryContextC &factory,const std::string &name,const TriggerC &trigger,SocketC::RefT &skt)
    {
      if(!factory.UseComponent(name,skt,false,typeid(ZmqN::SocketC)))
        return 0;
      return CallOnWrite(*skt,trigger);
    }


    //! Add a read trigger
    SocketDispatcherC::RefT ReactorC::CallOnRead(const SocketC &socket,const TriggerC &trigger)
    {
      SocketDispatcherC::RefT ret = new SocketDispatchTriggerC(socket,true,false,trigger);
      Add(*ret);
      return ret;
    }

    //! Add a write trigger
    SocketDispatcherC::RefT ReactorC::CallOnWrite(const SocketC &socket,const TriggerC &trigger)
    {
      SocketDispatcherC::RefT ret = new SocketDispatchTriggerC(socket,false,true,trigger);
      Add(*ret);
      return ret;
    }


    //! Add handler to system
    bool ReactorC::Add(const SocketDispatcherC &handler)
    {
      m_sockets.push_back(&handler);
      m_pollListChanged = true;
      return true;
    }

    //! Remove handler from system
    bool ReactorC::Remove(const SocketDispatcherC &handler)
    {
      for(unsigned i = 0;i < m_sockets.size();i++) {
        if(m_sockets[i] == &handler) {
          if(i < (m_sockets.size()-1))
            m_sockets[i] = m_sockets.back();
          m_sockets.pop_back();
          m_pollListChanged = true;
          return true;
        }
      }
      RavlAssertMsg(0,"Asked to remove unknown handler.");
      return false;
    }

    //! Run reactor loop.
    bool ReactorC::Run() {
      std::vector<zmq_pollitem_t> pollArr;


      // We keep an array of socket dispatchers we're using, 'inUse'
      // so that if a socket handler delete's itself from the reactor
      // it will be kept at least until the end of the poll cycle as
      // we don't want to delete a class we're calling.

      if(m_verbose) {
        RavlDebug("Starting reactor '%s' ",Name().data());
      }
      OnStart();

      std::vector<SocketDispatcherC::RefT> inUse;
      m_pollListChanged = true; // Make sure its refreshed!

      pollArr.reserve(m_sockets.size());
      zmq_pollitem_t *first = 0;
      while(!m_terminate) {
        if(m_pollListChanged) {
          pollArr.clear();
          inUse.clear();
          pollArr.reserve(m_sockets.size());
          inUse.reserve(m_sockets.size());
          zmq_pollitem_t item;
          for(unsigned i = 0;i < m_sockets.size();i++) {
            if(m_sockets[i]->SetupPoll(item)) {
              inUse.push_back(m_sockets[i]);
              pollArr.push_back(item);
            }
          }
          if(pollArr.size() > 0)
            first = &pollArr[0];
          else
            first = 0;
          m_pollListChanged = false;
        }
        double timeToNext = m_timedQueue.ProcessStep();
        if(timeToNext < 0 || (timeToNext > m_teminateCheckInterval && m_teminateCheckInterval > 0))
          timeToNext = m_teminateCheckInterval;
        long timeout = -1;
        if(m_teminateCheckInterval >= 0)
          timeout = timeToNext * 1000000.0;

        if(m_verbose) {
          RavlDebug("Reactor '%s' polling for %u sockets.",Name().data(),(unsigned) pollArr.size());
        }
        int ret = zmq_poll (first, pollArr.size(),timeout);
        if(m_verbose) {
          RavlDebug("Reactor '%s' got ready for %d sockets. (Timeout:%u) ",Name().data(),ret,timeout);
        }
        if(ret < 0) {
          int anErrno = zmq_errno ();
          // Shutting down ?
          if(anErrno == ETERM) {
            if(m_verbose) {
              RavlDebug("Reactor '%s' context shutdown.",Name().data());
            }
            break;
          }
          if(anErrno == EINTR) {
            if(m_verbose) {
              RavlDebug("Reactor '%s' Got interrupted.",Name().data());
            }
            continue;
          }
          RavlError("Reactor '%s' poll failed : %s ",Name().data(),zmq_strerror (anErrno));
          RavlAssertMsg(0,"unexpected error");
          continue;
        }
        unsigned i = 0;
        while(i < pollArr.size() && ret > 0) {
          // Avoid repeatedly setting up try/catch as it can be expensive.
          try {
            for(;i < pollArr.size() && ret > 0;i++) {
              if(pollArr[i].revents != 0)
                inUse[i]->Dispatch();
            }
          } catch(std::exception &ex) {
            RavlError("Caught c++ exception %s : %s ",RavlN::TypeName(typeid(ex)),ex.what());
            RavlAssert(0);
            i++;
          } catch(RavlN::ExceptionC &ex) {
            RavlError("Caught Ravl exception %s : %s ",RavlN::TypeName(typeid(ex)),ex.what());
            ex.Dump(std::cerr);
            RavlAssert(0);
            i++;
          } catch(...) {
            // FIXME: Be more informative!
            RavlError("Caught unknown exception dispatching message. ");
            RavlAssert(0);
            i++; // Skip it an go to next.
          }
        }
      }

      OnFinish();

      if(m_verbose) {
        RavlDebug("Shutdown of reactor '%s' complete.",Name().data());
      }
      return true;
    }

    //! Called by the main loop when its first run.
    bool ReactorC::OnStart()
    {

      return true;
    }

    //! Called when loop is exiting.
    bool ReactorC::OnFinish()
    {

      return true;
    }

    //! Schedule event for running on the reactor thread
    // Thread safe.
    // Returns an ID for the event, which can
    // be used for cancelling it.
    UIntT ReactorC::Schedule(const TriggerC &se)
    {
      UIntT ret = m_timedQueue.Schedule(0,se);
      RavlN::MutexLockC lock(m_accessWakeup);
      m_wakeup->Send(*m_wakeMsg);
      return ret;
    }

    //! Schedule event for running after time 't' (in seconds).
    // Thread safe.
    // Returns an ID for the event, which can
    // be used for cancelling it.
    UIntT ReactorC::Schedule(RealT t,const TriggerC &se,float period)
    {
      UIntT ret = m_timedQueue.Schedule(t,se,period);
      RavlN::MutexLockC lock(m_accessWakeup);
      m_wakeup->Send(*m_wakeMsg);
      return ret;
    }

    //! Schedule event for running.
    // Thread safe.
    // Returns an ID for the event, which can
    // be used for cancelling it.
    UIntT ReactorC::Schedule(const DateC &at,const TriggerC &se,float period)
    {
      UIntT ret = m_timedQueue.Schedule(at,se,period);
      RavlN::MutexLockC lock(m_accessWakeup);
      m_wakeup->Send(*m_wakeMsg);
      return ret;
    }

    //! Schedule event for running periodically.
    // Thread safe.
    // Returns an ID for the event, which can
    // be used for cancelling it.
    UIntT ReactorC::SchedulePeriodic(const TriggerC &se,float period)
    {
      UIntT ret = m_timedQueue.SchedulePeriodic(se,period);
      RavlN::MutexLockC lock(m_accessWakeup);
      m_wakeup->Send(*m_wakeMsg);
      return ret;
    }

    //! Cancel pending event.
    // Will return TRUE if event in cancelled before
    // it was run.
    bool ReactorC::Cancel(UIntT eventID)
    {
      return m_timedQueue.Cancel(eventID);
    }



    //! Called when owner handles drop to zero.
    void ReactorC::ZeroOwners() {
      m_terminate = true;
      ServiceThreadC::ZeroOwners();
    }

    ServiceThreadC::RefT Reactor2ServiceThread(const ReactorC::RefT &reactor)
    { return reactor.BodyPtr(); }

    DP_REGISTER_CONVERSION(Reactor2ServiceThread,1.0);

    static XMLFactoryRegisterConvertC<ReactorC,ServiceThreadC> g_regiserBreeder("RavlN::ZmqN::ReactorC");

  }
}
