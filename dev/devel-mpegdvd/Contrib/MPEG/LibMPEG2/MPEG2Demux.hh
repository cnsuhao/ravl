// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_MPEG2DEMUX_HEADER
#define RAVL_MPEG2DEMUX_HEADER 1
//////////////////////////////////////////////////////////////////
//! rcsid = "$Id$"
//! lib = RavlLibMPEG2
//! author = "Warren Moore"

#include "Ravl/DP/StreamOp.hh"
#include "Ravl/SArray1d.hh"
#include "Ravl/DList.hh"

namespace RavlImageN
{

  using namespace RavlN;

  class MPEG2DemuxBodyC :
    public DPIStreamOpBodyC< ByteT, ByteT >
  {
  public:
    MPEG2DemuxBodyC(ByteT track);
    //: Constructor.
    
    ~MPEG2DemuxBodyC();
    //: Destructor.
    
    ByteT Get()
    {
      ByteT data;
      if(!Get(data))
        throw DataNotReadyC("Failed to get next byte.");
      return data;
    }
    //: Get next byte.
    
    bool Get(ByteT &data);
    //: Get next byte.
    
    IntT GetArray(SArray1dC<ByteT> &data);
    //: Get an array of bytes from stream.
    // returns the number of elements processed

    bool IsGetEOS() const;
    //: Is it the EOS

    virtual bool GetAttr(const StringC &attrName, IntT &attrValue);
    //: Get a stream attribute.
    // Returns false if the attribute name is unknown.

  private:
    void BuildAttributes();
    //: Register stream attributes

  private:
    ByteT m_track;                                                          // Selected track to extract
    IntT m_state;                                                           // Current demultiplexer state
    IntT m_stateBytes;                                                      // Current number of bytes in buffer
    ByteT m_headBuf[264];                                                   // Demultiplexer head buffer
    
    SArray1dC< ByteT > m_dataIn;                                            // Input data buffer
    DListC< SArray1dC< ByteT > > m_dataOut;                                 // Output data buffer
    ByteT *m_bufStart, *m_bufEnd;                                           // Input data buffer pointers
  };

  class MPEG2DemuxC :
    public DPIStreamOpC< ByteT, ByteT >
  {
  public:
    MPEG2DemuxC() :
      DPEntityC(true)
    {}
    //: Default constructor.
    // Creates an invalid handle.

    MPEG2DemuxC(ByteT track) :
      DPEntityC(*new MPEG2DemuxBodyC(track))
    {}
    //: Constructor.
  };

}

#endif // RAVL_MPEG2DEMUX_HEADER
