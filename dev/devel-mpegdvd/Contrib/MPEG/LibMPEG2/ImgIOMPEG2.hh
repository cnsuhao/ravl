// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_IMGIOMPEG2_HEADER
#define RAVL_IMGIOMPEG2_HEADER 1
//////////////////////////////////////////////////////////////////
//! rcsid = "$Id$"
//! lib = RavlLibMPEG2
//! author = "Warren Moore"

#include "Ravl/DP/StreamOp.hh"
#include "Ravl/Image/Image.hh"
#include "Ravl/Image/ByteRGBValue.hh"
#include "Ravl/Stream.hh"
#include "Ravl/Hash.hh"
#include "Ravl/AVLTree.hh"
#include "Ravl/Tuple2.hh"
#include "Ravl/DList.hh"
#include "Ravl/Cache.hh"
#include "Ravl/Tuple2.hh"
#include "Ravl/DP/SPort.hh"

extern "C"
{
  typedef struct mpeg2dec_s mpeg2dec_t;  
  typedef unsigned int uint32_t;
  typedef unsigned char uint8_t;
  
  #include <mpeg2dec/mpeg2.h>
}

namespace RavlImageN
{
  
  class ImgILibMPEG2BodyC :
    public DPIStreamOpBodyC< ByteT, ImageC<ByteRGBValueC> >,
    public DPSeekCtrlBodyC
  {
  public:
    ImgILibMPEG2BodyC();
    //: Constructor.
    
    ~ImgILibMPEG2BodyC();
    //: Destructor.
    
    virtual ImageC<ByteRGBValueC> Get()
    {
      ImageC<ByteRGBValueC> img;
      if(!Get(img))
        throw DataNotReadyC("Failed to get next frame. ");
      return img;
    }
    //: Get next frame.
    
    virtual bool Get(ImageC<ByteRGBValueC> &img);
    //: Get next frame.
    
    virtual UIntT Tell() const
    { return m_frameNo; }
    //: Find current location in stream.
    
    virtual bool Seek(UIntT off);
    //: Seek to location in stream.
    
    virtual UIntT Size() const;
    //: Get the size of the file in frames (-1 if not known)
    
    virtual StreamPosT Tell64() const
    { return m_frameNo; }
    //: Find current location in stream.
    
    virtual bool Seek64(StreamPosT off);
    //: Seek to location in stream.
    
    virtual StreamPosT Size64() const;
    //: Get the size of the file in frames (-1 if not known)
    
    virtual bool IsGetEOS() const;
    //: Is it the EOS

    virtual bool GetAttr(const StringC &attrName,StringC &attrValue);
    //: Get a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.
    
  protected:
    virtual bool InitialSeek();
    //: Store the initial stream position
    
    virtual bool ReadData();
    //: Read data from stream into buffer.
    
    virtual bool DecodeGOP(UIntT firstFrameNo);
    //: Decode a whole GOP and put it in the image cache.
    
    virtual bool Decode(UIntT &frameNo, const mpeg2_info_t *info);
    //: Decode a block of data
    
    virtual void OnEnd();
    //: Called when a sequence end tag found
    
    virtual void BuildAttributes();
    //: Register stream attributes

  protected:
    mpeg2dec_t *m_decoder;                                                  // Decoder object
    IntT m_state;                                                           // Current decoder state
    SArray1dC<ByteT> m_buffer;                                              // Data buffer for decoder
    
    ByteT *m_bufStart, *m_bufEnd;                                           // Data buffer pointers
    Index2dC m_imgSize;                                                     // Frame size
    StreamPosT m_frameNo;                                                   // Desired seek frame
    StreamPosT m_lastRead;                                                  // Stream position at end of last read
    AVLTreeC<StreamPosT, StreamPosT> m_offsets;                             // Offsets of GOP in streams
    CacheC<StreamPosT,Tuple2C<ImageC<ByteRGBValueC>,IntT> > m_imageCache;   // Frame cache
    bool m_sequenceInit;                                                    // Sequence initialised indicator
    IntT m_lastFrameType;                                                   // Last decoded frame tpye indicator
    
    UIntT m_gopCount;                                                       // Cache GOP counter
    UIntT m_gopLimit;                                                       // Cache GOP max
    UIntT m_previousGop;                                                    // Frame number at previous GOP
    bool m_gopDone;                                                         // GOP read indicator
    bool m_endFound;                                                        // END found indicator
  };

  class ImgILibMPEG2C :
    public DPIStreamOpC< ByteT, ImageC<ByteRGBValueC> >,
    public DPSeekCtrlC
  {
  public:
    ImgILibMPEG2C() :
      DPEntityC(true)
    {}
    //: Default constructor.
    // Creates an invalid handle.

    ImgILibMPEG2C(bool) :
      DPEntityC(*new ImgILibMPEG2BodyC())
    {}
    //: Constructor.
    
  };
}

#endif
