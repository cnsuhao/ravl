// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2005, Omniperception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_FFMPEG_AVFORMAT_HEADER 
#define RAVL_FFMPEG_AVFORMAT_HEADER 
//! rcsid="$Id$"
//! lib=RavlLibFFmpeg

#include "Ravl/DP/SPort.hh"
#include "Ravl/Image/FFmpegPacket.hh"

#include <ffmpeg/avformat.h>

namespace RavlN {
  
  
  //: FFmpeg packet stream.
  
  class FFmpegPacketStreamBodyC 
    : public DPISPortBodyC<FFmpegPacketC>
  {
  public:
    FFmpegPacketStreamBodyC(const StringC &filename);
    //: Constructor
    
    FFmpegPacketStreamBodyC();
    //: Default constructor.
    
    ~FFmpegPacketStreamBodyC();
    //: Destructor.
    
    bool Open(const StringC &filename);
    //: Open file.
    
    bool CheckForVideo();
    //: Check for a readable video stream.
    
    AVFormatContext *FormatCtx()
    { return pFormatCtx; }
    //: Access format context.
    
    bool FirstVideoStream(IntT &videoStreamId,IntT &codecId);
    //: Find info about first video stream.
    
    virtual FFmpegPacketC Get();
    //: Get a packet from the stream.
    
    virtual bool Get(FFmpegPacketC &packet);
    //: Get a packet from the stream.
    
    virtual bool IsGetReady() const;
    //: Is get ready ?
    
    virtual bool IsGetEOS() const;
    //: End of stream ?
    
  protected:
    StringC filename;
    AVFormatContext *pFormatCtx;
  };
  
  //! userlevel=Normal
  //: FFmpeg packet stream. 
  //!cwiz:author
  
  class FFmpegPacketStreamC
    : public DPISPortC<FFmpegPacketC>
  {
  public:
    FFmpegPacketStreamC(const StringC & filename) 
      : DPEntityC(*new FFmpegPacketStreamBodyC(filename))
    {}
    //: Constructor 
    //!cwiz:author
    
    FFmpegPacketStreamC(bool)
      : DPEntityC(*new FFmpegPacketStreamBodyC())
    {}
    //: Constructor.
    
    FFmpegPacketStreamC()
      : DPEntityC(true)
    {}
    //: Default constructor.
    
    FFmpegPacketStreamC(const DPISPortC<FFmpegPacketC> &other)
      : DPEntityC(dynamic_cast<const FFmpegPacketStreamBodyC *>(other.BodyPtr(other)))
    {}
    //: Upcast constructor.
    
    bool Open(const StringC & filename) 
    { return Body().Open(filename); }
    //: Open file. 
    //!cwiz:author
    
    bool CheckForVideo() 
    { return Body().CheckForVideo(); }
    //: Check for a readable video stream. 
    //!cwiz:author

    AVFormatContext *FormatCtx()
    { return Body().FormatCtx(); }
    //: Access format context.
    
    bool FirstVideoStream(IntT &videoStreamId,IntT &codecId)
    { return Body().FirstVideoStream(videoStreamId,codecId); }
    //: Find info about first video stream.
    
  protected:
    FFmpegPacketStreamC(FFmpegPacketStreamBodyC &bod)
      : DPEntityC(bod)
    {}
    //: Body constructor. 
    
    FFmpegPacketStreamBodyC& Body()
    { return dynamic_cast<FFmpegPacketStreamBodyC &>(DPEntityC::Body()); }
    //: Body Access. 
    
    const FFmpegPacketStreamBodyC& Body() const
    { return dynamic_cast<const FFmpegPacketStreamBodyC &>(DPEntityC::Body()); }
    //: Body Access. 
    
  };
}


#endif

