// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_IMGIOMPEG2_HEADER
#define RAVL_IMGIOMPEG2_HEADER 1
//! rcsid="$Id$"
//! lib=RavlLibMPEG2
//! author="Charles Galambos"

#include "Ravl/DP/SPort.hh"
#include "Ravl/Image/Image.hh"
#include "Ravl/Image/ByteRGBValue.hh"
#include "Ravl/Stream.hh"
#include "Ravl/Hash.hh"
#include "Ravl/AVLTree.hh"
#include "Ravl/Tuple2.hh"
#include "Ravl/DList.hh"
#include "Ravl/Cache.hh"

extern "C" {
  typedef struct mpeg2dec_s mpeg2dec_t;  
}

namespace RavlN {
  class BitIStreamC;
}
namespace RavlImageN {
  
  
  class ImgILibMPEG2BodyC
    : public DPISPortBodyC<ImageC<ByteRGBValueC> >
  {
  public:
    ImgILibMPEG2BodyC(IStreamC &strm);
    //: Constructor.
    
    ~ImgILibMPEG2BodyC();
    //: Destructor.
    
    bool GetFrame(ImageC<ByteRGBValueC> &img);
    //: Get a single frame from the stream.
    
    bool DecodeGOP(UIntT firstFrameNo);
    //: Decode a whole GOP and put it in the image cache.
    
    bool BuildIndex(UIntT targetFrame);
    //: Build GOP index to frame.
    
    bool SkipToStartCode(BitIStreamC &is); 
    //: Skip to next start code.
    
    virtual ImageC<ByteRGBValueC> Get() {
      ImageC<ByteRGBValueC> img;
      if(!Get(img))
	throw DataNotReadyC("Failed to get next frame. ");
      return img;
    }
    //: Get next frame.
    
    virtual bool Get(ImageC<ByteRGBValueC> &img);
    //: Get next frame.
    
    virtual UIntT Tell() const
    { return frameNo; }
    //: Find current location in stream.
    
    virtual bool Seek(UIntT off);
    //: Seek to location in stream.
    
    virtual bool GetAttr(const StringC &attrName,IntT &attrValue);
    //: Get a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.
    
    virtual bool GetAttrList(DListC<StringC> &list) const;
    //: Get list of attributes available.
    // This method will ADD all available attribute names to 'list'.

  protected:
    bool ReadData();
    //: Read data from stream into buffer.
    
    IStreamC ins;
    mpeg2dec_t *decoder;
    IntT state;
    SArray1dC<ByteT> buffer;
    
    ByteT *bufStart, *bufEnd;
    Index2dC imgSize;
    UIntT allocFrameId;
    UIntT frameNo;
    UIntT maxFrameIndex;
    streampos lastRead;
    HashC<UIntT,ImageC<ByteRGBValueC> > images;
    AVLTreeC<UIntT,std::streampos> offsets;
    CacheC<UIntT,Tuple2C<ImageC<ByteRGBValueC>,IntT> > imageCache;
    bool sequenceInit;
    IntT lastFrameType;
  };

  class ImgILibMPEG2C
    : public DPISPortC<ImageC<ByteRGBValueC> >
  {
  public:
    ImgILibMPEG2C()
      : DPEntityC(true)
    {}
    //: Default constructor.
    // Creates an invalid handle.

    ImgILibMPEG2C(IStreamC &strm)
      : DPEntityC(*new ImgILibMPEG2BodyC(strm))
    {}
    //: Constructor.
    
  };
  
}


#endif
