// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlLibMPEG2
//! author="Charles Galambos"

#include "Ravl/Image/ImgIOMPEG2.hh"
#include "Ravl/IO.hh"
#include "Ravl/BitStream.hh"

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
  
  static char frameTypes[5] = { 'X','I', 'P','B','D' };
  
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
      maxFrameIndex(0),
      lastRead(0),
      offsets(true),
      imageCache(40),
      sequenceInit(false),
      lastFrameType(0)
  {
    decoder = mpeg2_init ();
    offsets.Insert(0,strm.Tell()); // Store initial offset.
    //BuildIndex(1);
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

  //: Build GOP index to frame.
  
  bool ImgILibMPEG2BodyC::BuildIndex(UIntT targetFrame) {
    cerr << "ImgILibMPEG2BodyC::BuildIndex(), Called. \n";
    BitIStreamC strm(ins);
    bool tryAgain = false;
    while(1) {
      SkipToStartCode(strm);
      ByteT startCode = strm.NextAlignedByte();
      switch(startCode) {
      case 0: // Picture.
	ONDEBUG(cerr << "Picture. \n");
	//ReadPicture();
	break;
      case 0xb2: // User data start.
	ONDEBUG(cerr << "User data. \n");
	break;
      case 0xb3: // Sequence header.
	ONDEBUG(cerr << "Sequence header. \n");
	break;
      case 0xb4: // Sequence error.
	ONDEBUG(cerr << "Sequence error. \n");
	break;
      case 0xb5: { // Extention start.
	ONDEBUG(cerr << "Extention: ");
	ByteT extentionType = strm.ReadUInt(4);
	switch(extentionType) {
	case 1:
	  ONDEBUG(cerr << "Sequence. ");
	  break;
	case 2:
	  ONDEBUG(cerr << "Sequence Display. ");
	  break;
	case 3:
	  ONDEBUG(cerr << "Quantisation . ");
	  break;
	case 4:
	  ONDEBUG(cerr << "Copyright. ");
	  break;
	case 5:
	  ONDEBUG(cerr << "Sequence Scalable. ");
	  break;
	case 7:
	  ONDEBUG(cerr << "Picture Display.");
	  break;
	case 8:
	  ONDEBUG(cerr << "Picture Coding.");
	  break;
	case 9:
	  ONDEBUG(cerr << "Picture Spatial Scalable.");
	  break;
	case 10:
	  ONDEBUG(cerr << "Picture Temporal Scalable.");
	  break;
	default:
	  ONDEBUG(cerr << "Reserved. ");
	}
	} break;
      case 0xb7: // Sequence end.
	ONDEBUG(cerr << "Sequence end. \n");
	tryAgain = false;
	break;
      case 0xb8: // Group start code.
	ONDEBUG(cerr << "GOP start. \n");
	break;
      default:
	if(startCode < 0xaf) { // Slice start ?
	  //ONDEBUG(cerr << "Slice start. \n");
	  break;
	}
	if(startCode > 0xb9) { // System start code.
	  ONDEBUG(cerr << "System start code. \n");
	  tryAgain = true;
	  break;
	}
	ONDEBUG(cerr << "Unknown start code " << (IntT) startCode  <<" \n");
	break;
      }
      //if(!tryAgain)
      //break;
    }
    
    return true;
  }
  
  //: Skip to next start code.
  
  bool ImgILibMPEG2BodyC::SkipToStartCode(BitIStreamC &strm) {
    IntT state = 0;
    ByteT dat;
    while(strm.good()) {
      switch(state) {
      case 0:
	if(strm.NextAlignedByte() != 0)
	  continue;
      case 1:
	if(strm.NextAlignedByte() != 0) {
	  state = 0; // Back to the start.
	  continue;
	}
      case 2: 
	if((dat = strm.NextAlignedByte()) == 1) 
	  return true;
	if(dat == 0)
	  state = 1; // Back to state 1, we've found a zero.
	else
	  state = 0; // Back to looking for first 0.
      }
      // Back to looking for zeros.
    }
    
    return true;
  }
  
  //: Get a stream attribute.
  // Returns false if the attribute name is unknown.
  // This is for handling stream attributes such as frame rate, and compression ratios.
  
  bool ImgILibMPEG2BodyC::GetAttr(const StringC &attrName,IntT &attrValue) {
    if(attrName == "frametype") {
      attrValue = lastFrameType;
      return true; 
    }
    return DPPortBodyC::GetAttr(attrName,attrValue);
  }
  
  //: Get list of attributes available.
  // This method will ADD all available attribute names to 'list'.
  
  bool ImgILibMPEG2BodyC::GetAttrList(DListC<StringC> &list) const {
    list.InsLast(StringC("frametype"));
    return DPPortBodyC::GetAttrList(list);
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
    streampos preParsePos = 0;
    streampos postParsePos = 0;

#if 0
    decoder->shift = 0xffffff00;
    decoder->state = STATE_SLICE;
    decoder->chunk_start = decoder->chunk_ptr = decoder->chunk_buffer;
    decoder->action = mpeg2_seek_header;
    mpeg2_pts (decoder,0);
#endif
    decoder->shift = 0xffffff00;
    //decoder->action = mpeg2_seek_sequence;
    //decoder->code = 0xb4;
    decoder->first_decode_slice = 1;
    decoder->nb_decode_slices = 0xb0 - 1;
    //mpeg2dec->convert_id = NULL;

    
    do {
      if(state >= 0) {
	preParsePos = lastRead + (streampos) (decoder->buf_start - &(buffer[0]));
#if 1
	//RavlAssert(decoder->buf_start > &(buffer[0]));
	if(decoder->code == 0xB3) {
	  cerr << "************************************************* \n";
	  ONDEBUG(cerr << "Recording " << localFrameNo << " At=" << preParsePos << " \n");
	  offsets.Insert(localFrameNo,preParsePos,false);
	  if(localFrameNo > maxFrameIndex)
	    maxFrameIndex = localFrameNo;
	  if(gotFrames)
	    return true;
	}
#endif
      }
      
      state = mpeg2_parse (decoder);
      if(state != -1) {
	postParsePos = lastRead + (streampos) (decoder->buf_start - &(buffer[0]));
      }
      
      ONDEBUG(cerr << "state=" << state << " buf_start=" << preParsePos << "\n");
      //ONDEBUG(cerr << "Got state: " << state << "\n");
      switch(state) {
      case -1:   // Got to end of buffer ?
	//ONDEBUG(cerr << "Got -1.\n");
	if(!ReadData()) {
	  ONDEBUG(cerr << "Failed to read data..\n");
	  state = -2;
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
      } break;
      case STATE_GOP: {
	ONDEBUG(cerr << "Got STATE_GOP.  \n");
      } break;
      case STATE_PICTURE: {
	UIntT ptype = info->current_picture->flags & PIC_MASK_CODING_TYPE;
	ONDEBUG(cerr << "Got PICTURE. frameno " << allocFrameId <<  " Type=" << frameTypes[ptype] <<"\n");
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
	  lastFrameType = info->display_picture->flags & PIC_MASK_CODING_TYPE;
	  ONDEBUG(cerr << "frameid=" << frameid << " " << info->display_picture->temporal_reference << " Type=" << frameTypes[lastFrameType] <<"\n");
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
