// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
////////////////////////////////////////////////////
//! rcsid="$Id$"
//! lib=RavlThreads
//! file="Ravl/OS/Threads/Posix/MTLockImpl.cc"

#include "Ravl/Threads/RWLock.hh"
#include "Ravl/MTLocks.hh"
#include <iostream.h>

namespace RavlN {
  void IncPThreadSysDBLock()
  {}
  
  static const int noLocks = 2;
  
  RWLockC posixDBRWLock[4];
  
  static void CheckLockID(int &lockId) {
    if(lockId >= noLocks || lockId < 0) {
      cerr << "SysDBLockImpl.cc: Illegal lock selected " << lockId << "\n";
      lockId = 0;
    }
  }
  
  void DoMTSysDBReadLock(int lockId)  { 
    CheckLockID(lockId);
    posixDBRWLock[lockId].RdLock(); 
  }

  void DoMTSysDBWriteLock(int lockId) {
    CheckLockID(lockId);
    posixDBRWLock[lockId].WrLock(); 
  }

  void DoMTSysDBUnlockRd(int lockId) { 
    CheckLockID(lockId);
    posixDBRWLock[lockId].UnlockRd();
  }
  
  void DoMTSysDBUnlockWr(int lockId) { 
    CheckLockID(lockId);
    posixDBRWLock[lockId].UnlockWr(); 
  }
  
  int DoMTSysGetThreadID()  {
#if 0
    VThread *thrd = VThread::GetCurrentThread();
    if(thrd == 0)
      return 0; 
    return (int) thrd->GetID();
#else
    return -1;
#endif
  }
  
  
  // Install race resolution function into the refrence counting mechanism.
  
  class PThreadInitC {
  public:
    PThreadInitC() { 
      MTGetThreadID = DoMTSysGetThreadID;
      MTReadLock = DoMTSysDBReadLock;
      MTWriteLock = DoMTSysDBWriteLock;
      MTUnlockRd = DoMTSysDBUnlockRd;
      MTUnlockWr = DoMTSysDBUnlockWr;
    }
  };
  
  PThreadInitC doVThreadInitC;
  
}