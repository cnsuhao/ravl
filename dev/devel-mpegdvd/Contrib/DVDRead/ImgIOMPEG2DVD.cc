// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU
// General Public License (GPL). See the gpl.licence file for details or
// see http://www.gnu.org/copyleft/gpl.html
// file-header-ends-here
//////////////////////////////////////////////////////////////////
//! rcsid = "$Id$"
//! lib = RavlDVDRead
//! author = "Warren Moore"

#include "Ravl/ImgIOMPEG2DVD.hh"
#include "Ravl/Image/MPEG2Demux.hh"
#include "Ravl/Threads/Signal.hh"

#define DODEBUG 0

#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlImageN
{
  using namespace RavlN;
  
  ImgILibMPEG2DVDBodyC::ImgILibMPEG2DVDBodyC(MPEG2DemuxC &demux, DVDReadC &dvd) :
    ImgILibMPEG2BodyC(false),
    m_demux(demux),
    m_dvd(dvd)
  {
    RavlAssertMsg(m_demux.IsValid(), "ImgILibMPEG2DVDBodyC::ImgILibMPEG2DVDBodyC invalid MPEG2 demultiplexer");
    RavlAssertMsg(m_dvd.IsValid(), "ImgILibMPEG2DVDBodyC::ImgILibMPEG2DVDBodyC invalid DVD reader");

    // Attach the signals
    Signal0C signalFlush = m_dvd.SignalFlush();
    Connect(signalFlush, m_demux, &MPEG2DemuxC::Reset);
    ConnectRef(signalFlush, *this, &ImgILibMPEG2DVDBodyC::Reset);
  }
  
  bool ImgILibMPEG2DVDBodyC::Reset()
  {
    return ImgILibMPEG2BodyC::Reset();
  }
  
}

