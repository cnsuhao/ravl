// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlLibMPEG2


#include "Ravl/Image/ImgIOMPEG2.hh"
#include "Ravl/IO.hh"

extern "C" {
  typedef unsigned int uint32_t;
  typedef unsigned char uint8_t;
#include <mpeg2dec/mpeg2.h>
#include <mpeg2dec/convert.h>
#include <mpeg2dec/mpeg2_internal.h>

}

#define DODEBUG 1

#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlImageN {
  
  //: Default constructor.
  
  ImgILibMPEG2BodyC::ImgILibMPEG2BodyC(IStreamC &strm) 
    : ins(strm),
      decoder(0),
      state(-2),
      buffer(16384*2),
      bufStart(0),
      bufEnd(0),
      allocFrameId(0),
      frameNo(0),
      lastRead(0),
      offsets(true),
      imageCache(40),
      sequenceInit(false)
  {
    decoder = mpeg2_init ();
    offsets.Insert(0,strm.Tell()); // Store initial offset.
  }
  
  //: Destructor.
  
  ImgILibMPEG2BodyC::~ImgILibMPEG2BodyC()
  {
    if(decoder!= 0)
      mpeg2_close (decoder);
  }
  
  //: Read data from stream into buffer.
  
  bool ImgILibMPEG2BodyC::ReadData() {
    if(!ins)
      return false;
    bufStart = &(buffer[0]);
    lastRead = ins.Tell();
    ins.read((char *) bufStart,buffer.Size());
    UIntT len = ins.gcount();
    bufEnd = bufStart +len;
    mpeg2_buffer (decoder, bufStart, bufEnd);
    return true;
  }

  
  //: Decode a whole GOP and put it in the image cache.
  
  bool ImgILibMPEG2BodyC::DecodeGOP(UIntT firstFrameNo) {
    ONDEBUG(cerr << "ImgILibMPEG2BodyC::DecodeGOP() called LastState= " << state << "\n");
    
    streampos at;

    // Find the GOP before the required frame...
    // Bit of a hack but will do for now.
    while(!offsets.Find(firstFrameNo,at)) {
      if(firstFrameNo ==0)
	return false;
      firstFrameNo--;
    } 
    ONDEBUG(cerr << "ImgILibMPEG2BodyC::DecodeGOP() called Seeking to " << at << " for frame " << firstFrameNo  <<"\n");
    ins.Seek(at);
    UIntT localFrameNo = firstFrameNo;
    
    bool gotFrames = false;
    if(!ReadData()) {
      ONDEBUG(cerr << "Failed to read data..\n");
      return false;
    }
    const mpeg2_info_t *info = mpeg2_info (decoder);
    
    do {
      UIntT bufOffset1 = (decoder->buf_start - &(buffer[0]));
      state = mpeg2_parse (decoder);
      UIntT bufOffset2 = (decoder->buf_start - &(buffer[0]));
      ONDEBUG(cerr << "state=" << state << " buf_start=" << bufOffset1 << " " << bufOffset2 << "  Len=" << bufOffset2 - bufOffset1 << "\n");
      //ONDEBUG(cerr << "Got state: " << state << "\n");
      switch(state) {
      case -1:   // Got to end of buffer ?
	//ONDEBUG(cerr << "Got -1.\n");
	if(!ReadData()) {
	  ONDEBUG(cerr << "Failed to read data..\n");
	  return false;
	}
	break;
      case STATE_SEQUENCE_REPEATED: //ONDEBUG(cerr << "Got SEQUENCE_REPEATED.\n");
      case STATE_SEQUENCE: { 
	ONDEBUG(cerr << "Got SEQUENCE.\n");
	if(!sequenceInit) {
	  sequenceInit = true;
	  imgSize = Index2dC(info->sequence->height,info->sequence->width);
	  mpeg2_convert (decoder, convert_rgb24, NULL);
	  mpeg2_custom_fbuf (decoder, 1);
	  info = mpeg2_info (decoder);
	  cerr << "ImageSize=" << imgSize << "\n";
	}
	{ // Mark frame.
	  streampos at = lastRead + (streampos) (bufOffset2);
	  ONDEBUG(cerr << "Recording " << localFrameNo << " At=" << at << " \n");
	  offsets.Insert(localFrameNo,at,false);
	  if(gotFrames)
	    return true;
	}
      } break;
      case STATE_GOP: {
	ONDEBUG(cerr << "Got STATE_GOP.  \n");
      } break;
      case STATE_PICTURE: {
	ONDEBUG(cerr << "Got PICTURE. frameno " << allocFrameId <<  "\n");
	uint8_t * buf[3];
	UIntT frameid =allocFrameId++;
	void * id = (void *) frameid;
	ImageC<ByteRGBValueC> nimg(imgSize[0].V(),imgSize[1].V());
	images[frameid] = nimg;
	buf[0] = (uint8_t*) &(nimg[0][0]);
	buf[1] = 0;
	buf[2] = 0;
	mpeg2_set_buf (decoder, buf, id);
      } break;
      case STATE_PICTURE_2ND: ONDEBUG(cerr << "Got PICTURE_2ND.\n");
	break;
      case STATE_END:  ONDEBUG(cerr << "Got END.\n");
	break; 
      case STATE_SLICE: {
	ONDEBUG(cerr << "Got SLICE. \n");
	if(info->display_fbuf != 0) {
	  UIntT frameid = (UIntT) info->display_fbuf->id;
	  ONDEBUG(cerr << "frameid=" << frameid << " " << info->display_picture->temporal_reference << " \n");	  
	  imageCache.Insert(localFrameNo,images[frameid]);
	  images.Del(frameid);
	  localFrameNo++;
	  gotFrames = true;
	}
      } break;
      case STATE_INVALID:
	ONDEBUG(cerr << "Got INVALID state.\n");
	break;
      default:
	cerr << "ImgILibMPEG2BodyC::Decode(), Unexpected state " << state << "\n";
      }
    } while(1);

    return true;
  }
  
  //: Get next frame.
  
  bool ImgILibMPEG2BodyC::Get(ImageC<ByteRGBValueC> &img) {
    ONDEBUG(cerr << "ImgILibMPEG2BodyC::Get(). Called. \n");
    if(!imageCache.LookupR(frameNo,img)) {
      if(!DecodeGOP(frameNo)) {
	cerr << "ImgILibMPEG2BodyC::Get(), Failed to decode GOP. \n";
	return false;
      }
    }
    if(!imageCache.LookupR(frameNo,img)) {
      // Probably a seek forward in the file to a GOP we haven't seen yet...
      return false;
    }
    frameNo++;   
    return true;
  }
  
  //: Seek to location in stream.
  
  bool ImgILibMPEG2BodyC::Seek(UIntT off) {
    ONDEBUG(cerr << "ImgILibMPEG2BodyC::Seek(). Called. Dest=" << off << "\n");
    
    // Move the decoder to the next gop before changing the buffer.
    frameNo = off;
    return true;
  }

}
