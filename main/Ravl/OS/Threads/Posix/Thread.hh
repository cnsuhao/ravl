// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_THREADSTHREAD_HEADER
#define RAVL_THREADSTHREAD_HEADER 1
//////////////////////////////////////////////////////
//! rcsid="$Id$"
//! file="Ravl/OS/Threads/Posix/Thread.hh"
//! lib=RavlThreads
//! userlevel=Normal
//! docentry="Ravl.OS.Threads"
//! author="Charles Galambos"
//! date="02/07/1999"
//! example=exThread.cc

#include "Ravl/config.h"
#if !defined(__sgi__)
#define _GNU_SOURCE 1
#define _POSIX_SOURCE 1
#endif

//#if defined(__sol2__)
#if RAVL_HAVE_SIGNAL_H
#include <sys/signal.h>
#endif

#if RAVL_HAVE_POSIX_THREADS
#include <pthread.h>
#endif

#if RAVL_HAVE_WIN32_THREADS
#include <windows.h>
#endif

#include "Ravl/RCHandleV.hh"

namespace RavlN
{
  //! userlevel=Normal
  
  void OSYield();
  //: Yield control of processor
  // call if you wish a brief delay in execution.  
  // Particularly useful if you are forced to poll for an event.
   
  //! userlevel=Normal
  UIntT CurrentThreadID();
  //: Get ID of current running thread.
  
  extern void cancellationHandler(void *data);
  //! userlevel=Develop
  //: Called when a thread is cancelled.

#if RAVL_HAVE_POSIX_THREADS    
  void *StartThread(void *Data);
#endif
#if RAVL_HAVE_WIN32_THREADS
  DWORD WINAPI StartThread(LPVOID data);
#endif
  
  //! userlevel=Develop
  //: Thread body.
  
  class ThreadBodyC 
    : public RCBodyVC
  {
  public:
    ThreadBodyC();
    //: Default constructor.

    ~ThreadBodyC();
    //: Destructor.
    
    bool Execute();
    // Start thread running, use this command to start thread running
    // after it has been created.  This function can only be called 
    // once.
        
    void Terminate();
    //: Terminate thread.
    // This function is very dangerous. Stopping a thread whilst its
    // running is likely to cause memory leaks, and deadlocks. <br>
    // It is much better to have a thread check a flag (see ThreadEventC) 
    // and exit normally. <br>
    // If your going to use this method you should ensure that the thread
    // is not using reference counting, and has no resource locks at the
    // time this method is called. <br>
    // THEAD SAFE.
    
    bool SetPriority(int Pri);
    // Set the priority of the process
    // 0 to 32767, Higher faster.
    // THEAD SAFE.
    
    inline UIntT ID() const
    { return ((UIntT) threadID); }
    // Get a unique ID for this thread.
    // NB. An id may no be assigned to the thread until
    // after Execute() has been called.
    // THEAD SAFE.
    
  protected:
    virtual int Start(); 
    //: Called when thread started.  
    // This function should be overloaded
    // is a derived class.  It is called with the new thread after 
    // Execute() is called. 
    
    virtual int End();
    //: Called when thread terminates normally.  
    // Overloading this method
    // is optional.  There is no need to call this function 
    // directly. 

  protected:
    volatile bool terminatePending; // True if Terminate function has been called.
    
  private:
    void Startup();
    //: Start user code.
    
    void Cancel();
    //: Cancel thread.

#if RAVL_HAVE_POSIX_THREADS
    pthread_t threadID;
#endif
#if RAVL_HAVE_WIN32_THREADS
    HANDLE threadID;
#endif
    
    bool live; // Set to true after thread is created.
    
#if RAVL_HAVE_POSIX_THREADS    
    friend void *StartThread(void *Data);
#endif
#if RAVL_HAVE_WIN32_THREADS
    friend DWORD WINAPI StartThread(LPVOID data);
#endif
    friend void cancellationHandler(void *data);
  };
  
  //! userlevel=Advanced
  //: Handle to a thread.
  // In general it is not necessary to use this class directly;
  // it is better to use the LaunchThread(...) functions to start
  // a thread on a method. <br>
  // Note: The thread itself holds a reference count on the object.
  // This reference count is held until the thread terminates. 
  
  class ThreadC 
    : public RCHandleC<ThreadBodyC>
  {
  public:
    ThreadC()
    {}
    //: Default constructor.
    // Creates an invalid handle.
    
  protected:
    ThreadC(ThreadBodyC &bod)
      : RCHandleC<ThreadBodyC>(bod)
    {}
    //: Body constructor.
    
    ThreadBodyC &Body() 
    { return RCHandleC<ThreadBodyC>::Body(); }
    //: Access body.
    
    const ThreadBodyC &Body() const
    { return RCHandleC<ThreadBodyC>::Body(); }
    //: Access body.
    
  public:
    bool Execute()
    { return Body().Execute(); }
    //: Start thread running.
    // use this command to start thread running
    // after it has been created.  This function can only be called 
    // once.
    
    void Terminate()
    { Body().Terminate(); }
    //: Terminate thread.
    // This function is very dangerous. Stopping a thread whilst its
    // running is likely to cause memory leaks, and deadlocks. <br>
    // It is much better to have a thread check a flag (see ThreadEventC) 
    // and exit normally. <br>
    // If your going to use this method you should ensure that the thread
    // is not using reference counting, and has no resource locks at the
    // time this method is called. <br>
    // THEAD SAFE.
    
    bool SetPriority(int pri)
    { return Body().SetPriority(pri); }
    // Set the priority of the process
    // 0 to 32767, Higher faster.
    // THEAD SAFE.
    
    inline UIntT ID() const
    { return Body().ID(); }
    // Get a unique ID for this thread.
    // NB. An id may no be assigned to the thread until
    // after Execute() has been called.
    // THEAD SAFE.
    
  };

  UIntT ThisThreadID();
  //: Get ID for this thread.
  
}



#endif
