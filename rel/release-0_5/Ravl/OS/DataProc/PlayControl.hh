// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_DPPLAYCONTROL_HEADER
#define RAVL_DPPLAYCONTROL_HEADER 1
////////////////////////////////////////////////
//! file="Ravl/OS/DataProc/PlayControl.hh"
//! lib=RavlDPMT
//! author="Charles Galambos"
//! date="16/03/99"
//! docentry="Ravl.OS.Data Processing"
//! rcsid="$Id$"

#include "Ravl/Threads/Mutex.hh"
#include "Ravl/Threads/Semaphore.hh"
#include "Ravl/DP/StreamOp.hh"
#include "Ravl/DP/SPort.hh"

namespace RavlN {

  //! userlevel=Develop
  //: Base play control body class.
  
  class DPPlayControlBodyC
    : public virtual DPEntityBodyC
  {
  public:
    DPPlayControlBodyC();
    //: Default constructor.
    
    DPPlayControlBodyC(const DPSeekCtrlC &nCntrl,bool nPassEOS = true,UIntT nstart = ((UIntT) -1),UIntT nend = ((UIntT) -1));
    //: Constructor.
    
    ~DPPlayControlBodyC();
    //: Destructor.
    
    IntT Speed(IntT ninc);
    //: Set increments.
    // 1 == Normal play 
    // -1=Reverse play 
    // 2=Double speed forward etc..
    
    UIntT Tell() const;
    //: Where are we in the stream ?
    // This gives the number of the NEXT frame 
    // that will be processed.
    
    UIntT LastFrame() const 
    { return lastFrame; }
    //: Get number of last frame read.
    
    UIntT Size() const;
    //: How long is the stream ?
    // This is the number of frames in the sequence.
    
    bool Seek(UIntT pos);
    //: Seek to an absolute position in stream
    
    void Pause();
    //: Pause playing.
    
    void Continue();
    //: Continue playing.
    
    void Jog(IntT frames = 1);
    //: Move 'frames' forward or backward.
    
    UIntT FixedStart() const { return start; }
    //: Get start of sub sequence.
    
    UIntT FixedEnd() const { return end; }
    //: Get start of sub sequence.
    
    void ToBeginning();
    //: Goto beginning of sequence. (for GUI)
    
    void FastFwd();
    //: Fast forward. (for GUI)
    
    void PlayFwd();
    //: Play forward. (for GUI)
    
    void JogFwd();
    //: Go forward 1 frame. (for GUI)
    
    void Stop();
    //: Stop (for GUI)
    
    void JogBkw();
    //: Go backward 1 frame. (for GUI)
    
    void PlayBkw();
    //: Play backward. (for GUI)
    
    void FastBkw();
    //: Play backward. (for GUI)
    
    void ToEnd();
    //: Goto end of sequence. (for GUI)
    
    void SeekTo(UIntT loc);
    //: Goto position in sequence. (for GUI)
    
    void PlaySubSequence(int mode,UIntT startfrm,UIntT endfrm) { 
      playMode = mode; subStart = startfrm; subEnd = endfrm; 
      RavlAssert(subStart < subEnd);
    }
    //: Play a sub-sequence.
    // 0-Once through whole sequence.
    // 1-Once through sub seq. 
    // 2-Repeat 
    // 3-back and forth.
    
    IntT &SubSeqStart() 
    { return subStart; }
    //: Set sub sequence start.
    
    IntT &SubSeqEnd() 
    { return subEnd; }
    //: Set sub sequence start.
    
    IntT &SubSeqMode() 
    { return playMode; }
    //: Set sub sequence start.
    
  protected:  
    bool Open(const DPSeekCtrlC &nCntrl);
    //: Open new video stream.
    // This assumes the input stream is locked by the calling function.
    
    bool CheckUpdate();
    //: Check state of stream after get.
    // This assumes the input stream is locked by the calling function.
    
    MutexC access;     // Access control.
    IntT inc;          // Increments -ve and +ve  0==Stopped.  1=Normal play..
    bool ok;       // Are operations succeding?
    bool pause;    // Actively pause stream ?
    SemaphoreC sema;   // Stream pause semaphore.  
    DPSeekCtrlC ctrl;  // Seek control handle.
    bool passEOS;  // Pass along End Of Stream.
    UIntT start;       // Inital frame no.  (this is a valid frame. )
    UIntT end;         // final frame no.   (this is a valid frame. )
    
    IntT playMode;     // 0-Once through 1-Once through sub seq. 2-Repeat 3-back and forth.
    IntT subStart;       // Inital frame no of sub sequence.  (this is a valid frame. )
    IntT subEnd;         // final frame no of sub sequence.   (this is a valid frame. )
    bool doneRev;  // Use in palindrome mode.
    IntT at;           // Cache of input stream position.
    IntT lastFrame;  // Last frame displayed.
  };
  
  //! userlevel=Normal
  //: Base play control handle class.
  
  class DPPlayControlC
    : public virtual DPEntityC
  {
  public:
    DPPlayControlC()
      : DPEntityC(true)
    {}
    //: Default constructor.
    
  protected:
    DPPlayControlC(const DPPlayControlBodyC &bod);
    //: Body constructor.
    
    DPPlayControlBodyC &Body() 
    { return dynamic_cast<DPPlayControlBodyC &>(DPEntityC::Body()); }
    //: Access body.
    
    const DPPlayControlBodyC &Body() const
    { return dynamic_cast<const DPPlayControlBodyC &>(DPEntityC::Body()); }
    //: Access body.
    
  public:
    UIntT LastFrame() const 
    { return Body().LastFrame(); }
    //: Get number of last frame read.
    
    inline IntT Speed(IntT ninc) 
    { return Body().Speed(ninc); }
    //: Set increments.
    // 1 == Normal play 
    // -1=Reverse play 
    // 2=Double speed forward etc..
    
    inline UIntT Tell() const 
    { return Body().Tell(); }
    //: Where are we in the stream ?
    
    inline UIntT Size() const 
    { return Body().Size(); }
    //: How long is the stream ?
    
    inline void Pause()  
    { Body().Pause(); }
    //: Pause playing.
    
    inline void Continue()  
    { Body().Continue(); }
    //: Continue playing.
    
    inline void Jog(IntT frames = 1)  
    { Body().Jog(frames); }
    //: Move 'frames' forward or backward.
    
    inline bool Seek(UIntT pos) 
    { return Body().Seek(pos); }
    //: Seek to an absolute position in stream
    
    UIntT FixedStart() const 
    { return Body().FixedStart(); }
    //: Get start of sub sequence.
    
    UIntT FixedEnd() const
    { return Body().FixedEnd(); }
    //: Get start of sub sequence.
    
    void ToBeginning()
    { Body().ToBeginning(); }
    
    //: Goto beginning of sequence. (for GUI)
    
    void FastFwd()
    { Body().FastFwd(); }
    //: Fast forward. (for GUI)
    
    void PlayFwd()
    { Body().PlayFwd(); }
    //: Play forward. (for GUI)
    
    void Stop()
    { Body().Stop(); }
    //: Stop (for GUI)
    
    void PlayBkw()
    { Body().PlayBkw(); }
    //: Play backward. (for GUI)
    
    void FastBkw()
    { Body().FastBkw(); }
    //: Play backward. (for GUI)
    
    void ToEnd()
    { Body().ToEnd(); }
    //: Goto end of sequence. (for GUI)
    
    void SeekTo(UIntT loc)
    { Body().SeekTo(loc); }
    //: Goto position in sequence. (for GUI)
    
    void PlaySubSequence(int mode,UIntT startfrm,UIntT endfrm)
    { Body().PlaySubSequence(mode,startfrm,endfrm); }
    //: Play a sub-sequence.
    // 0-Once through whole sequence.
    // 1-Once through sub seq. 
    // 2-Repeat 
    // 3-back and forth.
    
    IntT &SubSeqStart() 
    { return Body().SubSeqStart(); }
    //: Set sub sequence start.
    
    IntT &SubSeqEnd() 
    { return Body().SubSeqEnd(); }
    //: Set sub sequence start.
    
    IntT &SubSeqMode() 
    { return Body().SubSeqMode(); }
    //: Set sub sequence start.
    
  };
  
  ////////////////////////////////////////
  //! userlevel=Develop
  //: Stream operation base class.
  
  template<class DataT>
  class DPIPlayControlBodyC 
    : public DPIPortBodyC<DataT>,
      public DPPlayControlBodyC
  {
  public:
    DPIPlayControlBodyC()
    {}
    //: Default constructor.
    
    DPIPlayControlBodyC(const DPISPortC<DataT> &nin,bool nPassEOS = true,UIntT nstart = ((UIntT) -1),UIntT nend = ((UIntT) -1))
      : DPPlayControlBodyC(nin,nPassEOS,nstart,nend),
	input(nin)
    {}
    //: Constructor.
    
    virtual DataT Get() { 
      if(pause)
	sema.Wait();
      MutexLockC lock(access);
      CheckUpdate();
      RavlAssert(input.IsValid());
      DataT dat = input.Get();  // Possible exception.
      lock.Unlock();
      return dat;
    }
    // Get next piece of data.
    
    virtual bool Get(DataT &buff) { 
      if(pause)
	sema.Wait();
      MutexLockC lock(access);
      RavlAssert(input.IsValid());
      CheckUpdate();
      if(!input.Get(buff)) {
	cerr << "DPIPlayControlBodyC::Get() ERROR: Failed, attempting to fudge stream position... \n";
	at--;
	return true;
      }
      return true;
      // Unlock here.
    }
    //: Try and get next piece of data.
    // If none, return false.
    // else put data into buff.  
    
    virtual bool IsGetReady() const {
      // This really should be false,
      // but this causes older code to exit in confusion.
      if(pause) 
	return true; 
      MutexLockC lock(access);
      RavlAssert(input.IsValid());
      return input.IsGetReady(); 
      // Unlock here.
    }
    //: Is some data ready ?
    // true = yes.
    
    virtual bool IsGetEOS() const {
      if(!passEOS)
	return false;
      MutexLockC lock(access);
      RavlAssert(input.IsValid());
      return input.IsGetEOS(); 
      // Unlock here.
    }
    //: Has the End Of Stream been reached ?
    // true = yes.
    
    bool Open(const DPISPortC<DataT> &nPort) {
      MutexLockC lock(access);
      if(!DPPlayControlBodyC::Open(nPort))
	return false;
      input = nPort;
      if(pause) {
	lock.Unlock();
	sema.Post(); // Display at least first frame.
      }
      return true;
    }
    //: Open new video stream.
    
  protected:
    DPISPortC<DataT> input; // Where to get data from.
    
  public:
    inline const DPISPortC<DataT> &Input() { return input; }
    // Access input port.
    
  }; 
  
  ///////////////////////////////////
  //! userlevel=Normal
  //: Stream operation handle class.
  
  template<class DataT>
  class DPIPlayControlC 
    : public DPIPortC<DataT>,
      public DPPlayControlC
  {
  public:
    DPIPlayControlC() 
      : DPEntityC(true)
    {}
    //: Default Constructor.
    
    DPIPlayControlC(const DPISPortC<DataT> &nin,bool nPassEOS = true,UIntT nstart = ((UIntT) -1),UIntT nend = ((UIntT) -1))
      : DPEntityC(*new DPIPlayControlBodyC<DataT>(nin,nPassEOS,nstart,nend))
    {}
    //: Constructor.
    
    DPIPlayControlC(const DPIPlayControlC<DataT> &oth) 
      : DPEntityC(oth)
    {}
    //: Copy Constructor.
    
  protected:
    DPIPlayControlC(const DPIPlayControlBodyC<DataT> &bod)
      : DPEntityC((DPIPortBodyC<DataT> &) bod)
    {}
    //: Body constructor.
    
    DPIPlayControlBodyC<DataT> &Body() 
    { return dynamic_cast<DPIPlayControlBodyC<DataT> &>(DPEntityC::Body()); }
    //: Access body.
    
    const DPIPlayControlBodyC<DataT> &Body() const
    { return dynamic_cast<const DPIPlayControlBodyC<DataT> &>(DPEntityC::Body()); }
    //: Access body.
    
  public:  
    inline const DPISPortC<DataT> &Input() 
    { return Body().Input(); }
  //: Access input port.

  inline bool Open(const DPISPortC<DataT> &nPort)
    { return Body().Open(nPort); }
    //: Open a new input.
    
  };
  
}

#endif