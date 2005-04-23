// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2005, Omniperception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlLibFFmpeg

#include "Ravl/Image/FFmpegPacketStream.hh"
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
    : pFormatCtx(0)
  { 
    if(!Open(filename))
      throw ExceptionOperationFailedC("Failed to open file. ");
  }
  
  //: Default constructor.
  
  FFmpegPacketStreamBodyC::FFmpegPacketStreamBodyC()
    : pFormatCtx(0)
  {}

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
      if(ret) return true;
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
    
    ONDEBUG(dump_format(pFormatCtx, 0, filename, false));
    ONDEBUG(cerr << "FFmpegPacketStreamBodyC::Open(" << filename << "), Completed ok. \n");
    return true;
  }
  
  //: Get a packet from the stream.
  
  FFmpegPacketC FFmpegPacketStreamBodyC::Get() {
    FFmpegPacketC newPacket(true);
    if(av_read_packet(pFormatCtx, &newPacket.Packet()) < 0)
      throw DataNotReadyC("No more packets to read. ");
    return newPacket;
  }
  
  //: Get a packet from the stream.
  
  bool FFmpegPacketStreamBodyC::Get(FFmpegPacketC &packet) {
    packet = FFmpegPacketC(true);
    return (av_read_packet(pFormatCtx, &packet.Packet()) >= 0);
  }
  
  //: Is get ready ?
  
  bool FFmpegPacketStreamBodyC::IsGetReady() const
  { return pFormatCtx != 0; }
  
  //: End of stream ?
  
  bool FFmpegPacketStreamBodyC::IsGetEOS() const
  { return pFormatCtx == 0; }
  
}
