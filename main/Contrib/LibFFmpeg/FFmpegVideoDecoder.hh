// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2005, Omniperception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_FFMPEGVIDEODECODER_HEADER
#define RAVL_FFMPEGVIDEODECODER_HEADER 1
//! rcsid="$Id$"
//! lib=RavlLibFFmpeg

#include "Ravl/Image/FFmpegPacketStream.hh"
#include "Ravl/Image/Image.hh"
#include "Ravl/Image/ByteRGBValue.hh"

namespace RavlN {
  using namespace RavlImageN;
  
  //! userlevel=Develop
  //: FFmpeg Video decoder.
  
  class FFmpegVideoDecoderBaseC
  {
  public:
    FFmpegVideoDecoderBaseC(DPISPortC<FFmpegPacketC> &packetStream,IntT videoStreamId,IntT codecId);
    //: Constructor.
    
    FFmpegVideoDecoderBaseC();
    //: Default constructor.
    
    ~FFmpegVideoDecoderBaseC();
    //: Destructor.
    
    bool Open(DPISPortC<FFmpegPacketC> &packetStream,IntT videoStreamId,IntT codecId);
    //: Open a stream.
    
    bool DecodeFrame();
    //: Decode the next frame.
    
    bool GetFrame(ImageC<ByteRGBValueC> &frame);
    //: Get a frame of video from stream.
    
  protected:
    IntT videoStreamId;             // Id of video stream we're currently decoding.
    AVCodecContext *pCodecCtx;      // Video codec.
    DPISPortC<FFmpegPacketC> input; // Input stream.
    AVFrame *pFrame;
    FFmpegPacketC packet;           // Current packet.
    int      bytesRemaining;
    uint8_t  *rawData;
  };
  


  template<class ImageT> 
  class ImgIOFFmpegBodyC
    : public DPISPortBodyC<ImageT>,
      public FFmpegVideoDecoderBaseC
  {
  public:
    ImgIOFFmpegBodyC() 
    {}
    //: Constructor.
    
    ImgIOFFmpegBodyC(DPISPortC<FFmpegPacketC> &packetStream,IntT videoStreamId,IntT codecId) 
      : FFmpegVideoDecoderBaseC(packetStream,videoStreamId,codecId)
    {}
    //: Constructor.
    
    virtual bool Get(ImageC<ByteRGBValueC> &buff)
    { return GetFrame(buff); }
    //: Get next image.

    virtual ImageC<ByteRGBValueC> Get() { 
      ImageC<ByteRGBValueC> buff;
      if(!GetFrame(buff))
        throw DataNotReadyC("Frame not read.");
      return buff;
    }
    //: Get next image.
    
    virtual bool IsGetReady() const 
    { if(!input.IsValid()) return false; return input.IsGetReady(); }
    //: Is some data ready ?
    // TRUE = yes.
    // Defaults to !IsGetEOS().
    
    virtual bool IsGetEOS() const
    { if(!input.IsValid()) return true; return input.IsGetEOS(); }
    //: Has the End Of Stream been reached ?
    // TRUE = yes.
    
#if 0
    virtual bool Seek(UIntT off);
    //: Seek to location in stream.
    // Returns FALSE, if seek failed. (Maybe because its
    // not implemented.)
    // if an error occurered (Seek returned False) then stream
    // position will not be changed.
    
    virtual bool DSeek(IntT off);
    //: Delta Seek, goto location relative to the current one.
    
    virtual UIntT Tell() const; 
    //: Find current location in stream.
    
    virtual UIntT Size() const; 
    //: Find the total size of the stream.
#endif
    
  protected:
    
  };

  template<class ImageT> 
  class ImgIOFFmpegC
    : public DPISPortC<ImageT>
  {
  public:
    ImgIOFFmpegC(DPISPortC<FFmpegPacketC> &packetStream,IntT videoStreamId,IntT codecId)
      : DPEntityC(*new ImgIOFFmpegBodyC<ImageT>(packetStream,videoStreamId,codecId))
    {}
    //: Constructor.
    
    ImgIOFFmpegC()
      : DPEntityC(true)
    {}
    //: Default constructor.
    
  protected:
    
  };
  
  
}

#endif
