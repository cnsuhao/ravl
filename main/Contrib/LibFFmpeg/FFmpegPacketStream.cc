// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2005, Omniperception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlLibFFmpeg

#include "Ravl/Image/FFmpegPacketStream.hh"
#include "Ravl/DP/AttributeValueTypes.hh"
#include "Ravl/Exception.hh"

#define DODEBUG 0

#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlN {
  


  //: Constructor
  
  FFmpegPacketStreamBodyC::FFmpegPacketStreamBodyC(const StringC &filename) 
    : pFormatCtx(0),
      positionRefStream(0),
      currentTimeStamp(0)
  { 
    Init();
    if(!Open(filename))
      throw ExceptionOperationFailedC("Failed to open file. ");
  }
  
  //: Default constructor.
  
  FFmpegPacketStreamBodyC::FFmpegPacketStreamBodyC()
    : pFormatCtx(0),
      positionRefStream(0),
      currentTimeStamp(0)
  { Init(); }
  
  //: Initalise attributes.
  
  void FFmpegPacketStreamBodyC::Init() {
    RegisterAttribute(AttributeTypeNumC<RealT>("framerate","Frame rate of video",true,false,0.0,1000.0,0.01,25));
    
    RegisterAttribute(AttributeTypeStringC("filename","Original filename of stream",true,false,""));
    RegisterAttribute(AttributeTypeStringC("title","Title of stream",true,false,""));
    RegisterAttribute(AttributeTypeStringC("author","Author",true,false,""));
    RegisterAttribute(AttributeTypeStringC("copyright","Copyright for material",true,false,""));
    RegisterAttribute(AttributeTypeStringC("comment","Comment",true,false,""));
    RegisterAttribute(AttributeTypeStringC("album","album",true,false,""));
  }
  
  //: Destructor.
  
  FFmpegPacketStreamBodyC::~FFmpegPacketStreamBodyC() {
    // Close the video file
    if(pFormatCtx != 0)
      av_close_input_file(pFormatCtx);    
  }

  //: Find info about first video stream.
  
  bool FFmpegPacketStreamBodyC::FirstVideoStream(IntT &videoStreamId,IntT &codecId) {
    
    // Find the first video stream
    for (IntT i = 0; i < pFormatCtx->nb_streams; i++) {
      if (pFormatCtx->streams[i]->codec.codec_type != CODEC_TYPE_VIDEO) 
        continue;
      
      // Get a pointer to the codec context for the video stream
      AVCodecContext *pCodecCtx = &pFormatCtx->streams[i]->codec;
      
      // Find the decoder for the video stream
      AVCodec *pCodec = avcodec_find_decoder(pCodecCtx->codec_id);
      if (pCodec == NULL) 
        continue;
      
      videoStreamId = i;
      codecId = pCodecCtx->codec_id;
      
      positionRefStream = videoStreamId;
      frameRate     = pFormatCtx->streams[positionRefStream]->r_frame_rate;
      frameRateBase = pFormatCtx->streams[positionRefStream]->r_frame_rate_base;
      if(frameRateBase == 1)
        frameRateBase = 1000;
      ONDEBUG(cerr << "FFmpegPacketStreamBodyC::FirstVideoStream, FrameRate=" << frameRate << " FrameRateBase=" << frameRateBase << " \n");
      return true;
    }
    
    return false;
  }
  
  
  //: Check for a readable video stream.
  
  bool FFmpegPacketStreamBodyC::CheckForVideo() {
    // Check the file is open!
    if(pFormatCtx == 0) {
      ONDEBUG(cerr << "FFmpegPacketStreamBodyC::CheckForVideo no stream." << endl);
      return false;
    }

    ONDEBUG(cerr << "FFmpegPacketStreamBodyC::CheckForVideo streams= "<< pFormatCtx->nb_streams << endl);
    
    // Find the first video stream
    for (IntT i = 0; i < pFormatCtx->nb_streams; i++) {
      if (pFormatCtx->streams[i]->codec.codec_type != CODEC_TYPE_VIDEO) 
        continue;
      
      // Get a pointer to the codec context for the video stream
      AVCodecContext *pCodecCtx = &pFormatCtx->streams[i]->codec;
      
      // Find the decoder for the video stream
      AVCodec *pCodec = avcodec_find_decoder(pCodecCtx->codec_id);
      if (pCodec == NULL) 
        continue;
      
      ONDEBUG(cerr << "FFmpegPacketStreamBodyC::CheckForVideo codec found(" << (pCodec->name != NULL ? pCodec->name : "NULL") << ")" << endl);
        
      // Inform the codec that we can handle truncated bitstreams
      // i.e. bitstreams where frame boundaries can fall in the middle of packets
      if (pCodec->capabilities & CODEC_CAP_TRUNCATED)
        pCodecCtx->flags |= CODEC_FLAG_TRUNCATED;
      bool ret = false;
      
      // Open codec
      if (avcodec_open(pCodecCtx, pCodec) >= 0) {
        ONDEBUG(cerr << "FFmpegPacketStreamBodyC::CheckForVideo codec constructed ok. " << endl);
        ret = true;
      }
      
      // Clean up codec
      avcodec_close(pCodecCtx);
      
      // Did we succeed ?
      if(ret) 
        return true;
    }
    
    return false;
  }
  
  
  //: Open file.
  
  bool FFmpegPacketStreamBodyC::Open(const StringC &filename) {
    ONDEBUG(cerr << "FFmpegPacketStreamBodyC::Open(" << filename << "), Called \n");
    
    // Open video file
    if (av_open_input_file(&pFormatCtx, filename, NULL, 0, NULL) != 0) {
      ONDEBUG(cerr << "FFmpegPacketStreamBodyC::Open(" << filename << "), Failed to open file. \n");
      return false;
    }
    
    // Retrieve stream information
    if (av_find_stream_info(pFormatCtx) < 0) {
      ONDEBUG(cerr << "FFmpegPacketStreamBodyC::Open(" << filename << "), Failed to find stream info. \n");
      return false;
    }
    
    // Setup some default values.
    if(pFormatCtx->nb_streams >= 1) {
      positionRefStream = 0;
      frameRate     = pFormatCtx->streams[positionRefStream]->r_frame_rate;
      frameRateBase = pFormatCtx->streams[positionRefStream]->r_frame_rate_base;
      if(frameRateBase == 1)
        frameRateBase = 1000;
      ONDEBUG(cerr << "FFmpegPacketStreamBodyC::Open, FrameRate=" << frameRate << " FrameRateBase=" << frameRateBase << " \n");
    }
      
    ONDEBUG(dump_format(pFormatCtx, 0, filename, false));
    ONDEBUG(cerr << "FFmpegPacketStreamBodyC::Open(" << filename << "), Completed ok. Streams=" << pFormatCtx->nb_streams << "\n");
    return true;
  }
  
  //: Get a packet from the stream.
  
  FFmpegPacketC FFmpegPacketStreamBodyC::Get() {
    FFmpegPacketC newPacket(true);
    if(av_read_packet(pFormatCtx, &newPacket.Packet()) < 0)
      throw DataNotReadyC("No more packets to read. ");
    
    if(newPacket.StreamIndex() == positionRefStream) {
      currentTimeStamp = newPacket.DecodeTimeStamp() + 1;
      ONDEBUG(cerr << "FFmpegPacketStreamBodyC::Get, currentTimeStamp=" << currentTimeStamp << "\n");
    }
    return newPacket;
  }
  
  //: Get a packet from the stream.
  
  bool FFmpegPacketStreamBodyC::Get(FFmpegPacketC &packet) {
    packet = FFmpegPacketC(true);
    if(av_read_packet(pFormatCtx, &packet.Packet()) < 0)
      return false;
    if(packet.StreamIndex() == positionRefStream) {
      currentTimeStamp = packet.DecodeTimeStamp() + 1;
      ONDEBUG(cerr << "FFmpegPacketStreamBodyC::Get, currentTimeStamp=" << currentTimeStamp << "\n");
    }
    return true;
  }
  
  //: Is get ready ?
  
  bool FFmpegPacketStreamBodyC::IsGetReady() const
  { return pFormatCtx != 0; }
  
  //: End of stream ?
  
  bool FFmpegPacketStreamBodyC::IsGetEOS() const
  { return pFormatCtx == 0; }
  
  //: Get a attribute.
  // Returns false if the attribute name is unknown.
  // This is for handling attributes such as frame rate, and compression ratios.
  
  bool FFmpegPacketStreamBodyC::GetAttr(const StringC &attrName,StringC &attrValue) {
    if(attrName=="duration") {
      if(pFormatCtx == 0) {
        attrValue = StringC();
        return true;
      }
      attrValue = StringC(pFormatCtx->duration);
      return true;
    }
    if(attrName=="filename") {
      if(pFormatCtx == 0) {
        attrValue = StringC();
        return true;
      }
      attrValue = pFormatCtx->filename;
      return true;
    }
    if(attrName=="title") {
      if(pFormatCtx == 0) {
        attrValue = StringC();
        return true;
      }
      attrValue = pFormatCtx->title;
      return true;
    }
    if(attrName=="author") {
      if(pFormatCtx == 0) {
        attrValue = StringC();
        return true;
      }
      attrValue = pFormatCtx->author;
      return true;
    }
    if(attrName=="copyright") {
      if(pFormatCtx == 0) {
        attrValue = StringC();
        return true;
      }
      attrValue = pFormatCtx->copyright;
      return true;
    }
    if(attrName=="comment") {
      if(pFormatCtx == 0) {
        attrValue = StringC();
        return true;
      }
      attrValue = pFormatCtx->comment;
      return true;
    }
    if(attrName=="album") {
      if(pFormatCtx == 0) {
        attrValue = StringC();
        return true;
      }
      attrValue = pFormatCtx->album;
      return true;
    }
    
    return DPISPortBodyC<FFmpegPacketC>::GetAttr(attrName,attrValue);
  }
  
  //: Get a attribute.
  // Returns false if the attribute name is unknown.
  // This is for handling attributes such as frame rate, and compression ratios.
  
  bool FFmpegPacketStreamBodyC::GetAttr(const StringC &attrName,IntT &attrValue) {
    return DPISPortBodyC<FFmpegPacketC>::GetAttr(attrName,attrValue);
  }
  
  //: Get a attribute.
  // Returns false if the attribute name is unknown.
  // This is for handling attributes such as frame rate, and compression ratios.
  
  bool FFmpegPacketStreamBodyC::GetAttr(const StringC &attrName,RealT &attrValue) {
    if(attrName == "framerate") {
      const AVStream *st = pFormatCtx->streams[positionRefStream];
      //cerr << "FFmpegPacketStreamBodyC::GetAttr Framerate = num=" << st->time_base.num << " den=" << st->time_base.den << "\n";
      if(st->time_base.num != 0)
        attrValue = (RealT) st->time_base.den / (RealT) st->time_base.num;
      else
        attrValue = (RealT) st->time_base.den;
      return true;
    }
    return DPISPortBodyC<FFmpegPacketC>::GetAttr(attrName,attrValue);
  }
  
  //: Get a attribute.
  // Returns false if the attribute name is unknown.
  // This is for handling attributes such as frame rate, and compression ratios.
  
  bool FFmpegPacketStreamBodyC::GetAttr(const StringC &attrName,bool &attrValue) {
    return DPISPortBodyC<FFmpegPacketC>::GetAttr(attrName,attrValue);
  }
  
  //: Seek to location in stream.
  // Returns FALSE, if seek failed. (Maybe because its
  // not implemented.)
  // if an error occurered (Seek returned False) then stream
  // position will not be changed.
  
  bool FFmpegPacketStreamBodyC::Seek(UIntT off) {
    return Seek64(off);
  }
  
  bool FFmpegPacketStreamBodyC::DSeek(IntT off) {
    return FFmpegPacketStreamBodyC::DSeek64(off);
  }
  
  //: Find current location in stream.
  
  UIntT FFmpegPacketStreamBodyC::Tell() const {
    return FFmpegPacketStreamBodyC::Tell64();
  }
  
  //: Find the total size of the stream.
  
  UIntT FFmpegPacketStreamBodyC::Size() const {
    return FFmpegPacketStreamBodyC::Size64();
  }
  
  //: Find current location in stream.
  
  Int64T FFmpegPacketStreamBodyC::Tell64() const {
    ONDEBUG(cerr << "FFmpegPacketStreamBodyC::Tell64 = " << currentTimeStamp << " \n");
    return currentTimeStamp;
  }
   
  //: Seek to location in stream.
  // Returns FALSE, if seek failed. (Maybe because its
  // not implemented.)
  // if an error occurered (Seek returned False) then stream
  // position will not be changed.
  
  bool FFmpegPacketStreamBodyC::DSeek64(Int64T off) {
    ONDEBUG(cerr << "FFmpegPacketStreamBodyC::DSeek64 Delta=" << off << " \n");
    return Seek64(off + currentTimeStamp);
  }
  
  //: Seek to location in stream.
  // Returns FALSE, if seek failed. (Maybe because its
  // not implemented.)
  // if an error occurered (Seek returned False) then stream
  // position will not be changed.
  
  bool FFmpegPacketStreamBodyC::Seek64(Int64T off) {
    if(pFormatCtx == 0)
      return false;
    ONDEBUG(cerr << "FFmpegPacketStreamBodyC::Seek64 to " << off << " \n");
    if(av_seek_frame(pFormatCtx, positionRefStream, Frame2Time(off)) < 0) {
      cerr << "FFmpegPacketStreamBodyC::Seek, Seek failed. " << off << " \n";
      return false;
    }
    currentTimeStamp = off;
    return true;
  }
  
  //: Find the total size of the stream.
  
  Int64T FFmpegPacketStreamBodyC::Size64() const {
    if(pFormatCtx == 0)
      return 0;
    ONDEBUG(cerr << "FFmpegPacketStreamBodyC::Size64, Duration=" << pFormatCtx->duration << " Start=" << pFormatCtx->start_time << " FrameRateBase=" << frameRateBase << "\n");
    return Time2Frame(pFormatCtx->duration + pFormatCtx->start_time);
  }
  
  //: Find the total size of the stream.
  
  Int64T FFmpegPacketStreamBodyC::Start64() const {
    if(pFormatCtx == 0)
      return 0;
    ONDEBUG(cerr << "FFmpegPacketStreamBodyC::Start64, " << pFormatCtx->start_time << " \n");
    return Frame2Time(pFormatCtx->start_time);
  }
  
  //: Find the total size of the stream.
  
  UIntT FFmpegPacketStreamBodyC::Start() const {
    return FFmpegPacketStreamBodyC::Start64();
  }
  
  //: Convert a frame no into a time
  
  Int64T FFmpegPacketStreamBodyC::Frame2Time(Int64T arg) const {
    const AVStream *st = pFormatCtx->streams[positionRefStream];
    return av_rescale(arg, AV_TIME_BASE * st->time_base.num, st->time_base.den);
  }
  
  //: Convert a  time into a frame no
  
  Int64T FFmpegPacketStreamBodyC::Time2Frame(Int64T arg) const {
    const AVStream *st = pFormatCtx->streams[positionRefStream];
    return av_rescale(arg, st->time_base.den, AV_TIME_BASE*(int64_t)st->time_base.num);
  }

}
