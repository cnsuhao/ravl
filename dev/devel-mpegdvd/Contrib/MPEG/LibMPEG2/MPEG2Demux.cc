// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU
// General Public License (GPL). See the gpl.licence file for details or
// see http://www.gnu.org/copyleft/gpl.html
// Demultiplexer based upon parts of mpeg2dec, which can be found at
// http://libmpeg2.sourceforge.net/
// file-header-ends-here
//////////////////////////////////////////////////////////////////
//! rcsid = "$Id$"
//! lib = RavlLibMPEG2
//! author = "Warren Moore"

#include "Ravl/Image/MPEG2Demux.hh"
#include "Ravl/DP/AttributeValueTypes.hh"

#define DODEBUG 0

#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

#define DEMUX_HEADER 0
#define DEMUX_DATA 1
#define DEMUX_SKIP 2

namespace RavlImageN
{
  
  using namespace RavlN;
  
  static const int mpeg1_skip_table[16] = { 0, 0, 4, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

  MPEG2DemuxBodyC::MPEG2DemuxBodyC(ByteT track) :
    m_track(track),
    m_state(DEMUX_SKIP),
    m_stateBytes(0)
  {
    cerr << "MPEG2DemuxBodyC::MPEG2DemuxBodyC" << endl;
    BuildAttributes();
  }
  
  MPEG2DemuxBodyC::~MPEG2DemuxBodyC()
  {
  }
  
  bool MPEG2DemuxBodyC::Get(ByteT &data)
  {
    // Check we're not at the end of the stream
    if (IsGetEOS())
      return false;
    
    SArray1dC<ByteT> dataOut(1);
    bool read = GetArray(dataOut) > 0;
    if (read)
      data = dataOut[0];
    return read;
  }
  
  IntT MPEG2DemuxBodyC::GetArray(SArray1dC<ByteT> &data)
  {
    cerr << "MPEG2DemuxBodyC::GetArray" << endl;
    return 0;
  }
  
  bool MPEG2DemuxBodyC::IsGetEOS() const
  {
    // Check the output buffer here first

    if (input.IsGetEOS())
    {
      ONDEBUG(cerr << "MPEG2DemuxBodyC::IsGetEOS input at EOS" << endl;)
      return true;
    }
    
    return false;
  }

  bool MPEG2DemuxBodyC::GetAttr(const StringC &attrName, IntT &attrValue)
  {
    if (attrName == "track")
    {
      attrValue = m_track;
      return true; 
    }
    return DPPortBodyC::GetAttr(attrName, attrValue);
  }

  void MPEG2DemuxBodyC::BuildAttributes()
  {
    RegisterAttribute(AttributeTypeNumC<ByteT>("track", "MPEG demultiplexed track", true, false));
  }

}

