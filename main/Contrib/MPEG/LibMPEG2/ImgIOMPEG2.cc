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
//! author = "Charles Galambos"

#include "Ravl/Image/ImgIOMPEG2.hh"
#include "Ravl/IO.hh"
#include "Ravl/BitStream.hh"
#include "Ravl/Array2dIter.hh"
#include "Ravl/Image/RealYUVValue.hh"
#include "Ravl/Image/ByteYUVValue.hh"
#include "Ravl/Image/RealRGBValue.hh"
#include "Ravl/Image/RGBcYUV.hh"
#include <fstream>

extern "C" {
#include <mpeg2dec/convert.h>
#include <mpeg2dec/mpeg2_internal.h>
}

#define DODEBUG 0

#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

#define USE_OWN_RGBCONV 1

#define DEMUX_HEADER 0
#define DEMUX_DATA 1
#define DEMUX_SKIP 2

namespace RavlImageN
{
  
#if DODEBUG
  static char frameTypes[5] = { 'X','I', 'P','B','D' };
#endif
  
  static int mpeg1_skip_table[16] = { 0, 0, 4, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

  //: Default constructor.
  
  ImgILibMPEG2BodyC::ImgILibMPEG2BodyC(IntT demuxTrack) :
    decoder(0),
    m_state(-2),
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
    lastFrameType(0),
    m_demuxTrack(demuxTrack),
    m_demuxState(DEMUX_SKIP),
    m_demuxStateBytes(0)
  {
    decoder = mpeg2_init();
    
    // Check the demux track
    if (m_demuxTrack > 0 && (m_demuxTrack < 0xe0 || m_demuxTrack > 0xef))
    {
      cerr << "ImgILibMPEG2BodyC::ImgILibMPEG2BodyC invalid demultiplex track.\n";
      m_demuxTrack = -1;
    }

//    BuildIndex(1);
  }
  
  //: Destructor.
  
  ImgILibMPEG2BodyC::~ImgILibMPEG2BodyC()
  {
    if(decoder != 0)
      mpeg2_close(decoder);
  }
  
  bool ImgILibMPEG2BodyC::InitialSeek()
  {
    RavlAssertMsg(offsets.IsEmpty(), "ImgILibMPEG2BodyC::InitialSeek called with non-empty offsets");
    //cerr << "ImgILibMPEG2BodyC::InitialSeek" << endl;
    if (input.IsValid())
    {
      DPSeekCtrlC seek(input);
      if (seek.IsValid())
      {
        offsets.Insert(0, seek.Tell64());
        
        return true;
      }
    }
    
    return false;
  }

  //: Read data from stream into buffer.
  
  bool ImgILibMPEG2BodyC::ReadData()
  {
    RavlAssertMsg(input.IsValid(), "ImgILibMPEG2BodyC::ReadData called with invalid input");
    
    if(input.IsGetEOS())
      return false;
    
    DPSeekCtrlC seek(input);
    if (!seek.IsValid())
      return false;
    
    bufStart = &(buffer[0]);
    lastRead = seek.Tell64();
    UIntT len = input.GetArray(buffer);
    bufEnd = bufStart + len;
    return true;
  }

  //: Get a stream attribute.
  // Returns false if the attribute name is unknown.
  // This is for handling stream attributes such as frame rate, and compression ratios.
  
  bool ImgILibMPEG2BodyC::GetAttr(const StringC &attrName,IntT &attrValue)
  {
    if(attrName == "frametype") {
      attrValue = lastFrameType;
      return true; 
    }
    return DPPortBodyC::GetAttr(attrName,attrValue);
  }
  
  //: Get list of attributes available.
  // This method will ADD all available attribute names to 'list'.
  
  bool ImgILibMPEG2BodyC::GetAttrList(DListC<StringC> &list) const
  {
    list.InsLast(StringC("frametype"));
    return DPPortBodyC::GetAttrList(list);
  }

  //: Decode a whole GOP and put it in the image cache.
  
  bool ImgILibMPEG2BodyC::DecodeGOP(UIntT firstFrameNo)
  {
    ONDEBUG(cerr << "ImgILibMPEG2BodyC::DecodeGOP called last state= " << m_state << "\n");
    
    // If not set, get the initial seek position
    if (offsets.IsEmpty() && !InitialSeek())
    {
      cerr << "ImgILibMPEG2BodyC::DecodeGOP unable to find initial seek position" << "\n";
      return false;
    }
    
    // Find the GOP before the required frame...
    // Bit of a hack but will do for now.
    StreamPosT at;
    while(!offsets.Find(firstFrameNo, at))
    {
      if(firstFrameNo == 0)
        return false;
      firstFrameNo--;
    }
    ONDEBUG(cerr << "ImgILibMPEG2BodyC::DecodeGOP seeking to " << at << " for frame " << firstFrameNo  <<"\n");
    
    DPSeekCtrlC seek(input);
    if (!seek.IsValid())
      return false;
    seek.Seek64(at);
    
    UIntT localFrameNo = firstFrameNo;
    bool gotFrames = false;
    UIntT preParsePos = 0;

    if(ReadData())
      mpeg2_buffer(decoder, bufStart, bufEnd);
    else
      return false;
    const mpeg2_info_t *info = mpeg2_info(decoder);

    /*
    decoder->shift = 0xffffff00;
    decoder->state = STATE_SLICE;
    decoder->chunk_start = decoder->chunk_ptr = decoder->chunk_buffer;
    decoder->action = mpeg2_seek_header;
    mpeg2_pts (decoder,0);
    */
    
    decoder->shift = 0xffffff00;
    //decoder->action = mpeg2_seek_sequence;
    //decoder->code = 0xb4;
    decoder->first_decode_slice = 1;
    decoder->nb_decode_slices = 0xb0 - 1;
    //mpeg2dec->convert_id = NULL;

    do
    {
      m_state = mpeg2_parse(decoder);
      if (m_state >= 0)
      {
        preParsePos = lastRead + (UIntT) (decoder->buf_start - &(buffer[0]));
        if(decoder->code == 0xB3) {
          ONDEBUG(cerr << "ImgILibMPEG2BodyC::DecodeGOP ******** Recording " << localFrameNo << " at " << preParsePos << " \n");
          offsets.Insert(localFrameNo, preParsePos, false);
          if(localFrameNo > maxFrameIndex)
            maxFrameIndex = localFrameNo;
          if(gotFrames)
            return true;
        }
      }

      if (m_state == -1)
      {
        if(ReadData())
          mpeg2_buffer(decoder, bufStart, bufEnd);
        else
        {
          ONDEBUG(cerr << "ImgILibMPEG2BodyC::DecodeGOP failed to read data..\n");
          m_state = -2;
          return false;
        }
      }

      ONDEBUG(cerr << "ImgILibMPEG2BodyC::DecodeGOP state=" << m_state << " preParsePos=" << preParsePos << "\n");
      Decode(localFrameNo, info, gotFrames);
    } while(true);

    return true;
  }
  
  
  bool ImgILibMPEG2BodyC::DemultiplexGOP(UIntT firstFrameNo)
  {
    ONDEBUG(cerr << "ImgILibMPEG2BodyC::DemultiplexGOP called last state= " << m_state << "\n");
    
    // If not set, get the initial seek position
    if (offsets.IsEmpty() && !InitialSeek())
    {
      cerr << "ImgILibMPEG2BodyC::DecodeGOP unable to find initial seek position" << "\n";
      return false;
    }
    
    // Find the GOP before the required frame...
    // Bit of a hack but will do for now.
    StreamPosT at;
    while(!offsets.Find(firstFrameNo,at))
    {
      if(firstFrameNo ==0)
        return false;
      firstFrameNo--;
    } 
    ONDEBUG(cerr << "ImgILibMPEG2BodyC::DemultiplexGOP seeking to " << at << " for frame " << firstFrameNo  <<"\n");
    
    DPSeekCtrlC seek(input);
    if (!seek.IsValid())
      return false;
    seek.Seek64(at);
    
    // Reset the decoder if the first frame of the GOP is zero.
    // TODO: This doesn't actually work properly!
    if (firstFrameNo == 0)
    {
      // Close the decoder
      mpeg2_close(decoder);
      
      // Restart the decoder
      decoder = mpeg2_init();
      m_state = -2;
      sequenceInit = false;
      m_demuxState = DEMUX_SKIP;
      m_demuxStateBytes = 0;
    }
    else
    {
      // TODO: This initial read seems to cover over the problems with back seeking :o)
      if(!ReadData())
      {
        ONDEBUG(cerr << "ImgILibMPEG2BodyC::DemultiplexGOP failed to read data.\n");
        return false;
      }
    }

    UIntT localFrameNo = firstFrameNo;
    bool gotFrames = false;
    m_gopCount = 0;
    const mpeg2_info_t *info = mpeg2_info(decoder);

    do
    {
      if(m_gopCount > 1 && gotFrames) {
        ONDEBUG(cerr << "ImgILibMPEG2BodyC::DemultiplexGOP ******** Recording " << localFrameNo << " at " << lastRead << " \n");
        offsets.Insert(localFrameNo, lastRead, false);
        if(localFrameNo > maxFrameIndex)
          maxFrameIndex = localFrameNo;
        break;
      }
    
      if(!ReadData()) {
        ONDEBUG(cerr << "ImgILibMPEG2BodyC::DemultiplexGOP failed to read data.\n");
        m_state = -2;
        return false;
      }
      
      Demultiplex(localFrameNo, info, gotFrames);
      
    } while(true);

    return true;
  }
  
  bool ImgILibMPEG2BodyC::Decode(UIntT &frameNo, const mpeg2_info_t *info, bool &gotFrames)
  {
//    ONDEBUG(cerr << "ImgILibMPEG2BodyC::Decode state=" << m_state << "\n");

    switch(m_state)
    {
    case -1:   // Got to end of buffer ?
      //        ONDEBUG(cerr << "ImgILibMPEG2BodyC::Decode need input\n";)
      break;
        
    case STATE_SEQUENCE_REPEATED:
    case STATE_SEQUENCE:{ 
      ONDEBUG(cerr << "ImgILibMPEG2BodyC::Decode got SEQUENCE.\n");
      if(!sequenceInit) {
	sequenceInit = true;
	imgSize = Index2dC(info->sequence->height,info->sequence->width);
#if USE_OWN_RGBCONV
	//mpeg2_convert(decoder, my_convert_rgb , NULL);
#else
	//mpeg2_convert(decoder, convert_rgb24, NULL);
#endif
	//mpeg2_custom_fbuf(decoder, 1);
	info = mpeg2_info(decoder);
	ONDEBUG(cerr << "ImgILibMPEG2BodyC::Decode imageSize=" << imgSize << "\n";)
	  }
    } break;
      
    case STATE_GOP:
      ONDEBUG(cerr << "ImgILibMPEG2BodyC::Decode got STATE_GOP.\n");
      m_gopCount++;
      break;
	
    case STATE_PICTURE: {
#if 0	
      ONDEBUG(cerr << "ImgILibMPEG2BodyC::Decode got PICTURE.\n");
      ONDEBUG(UIntT ptype = info->current_picture->flags & PIC_MASK_CODING_TYPE);
      ONDEBUG(cerr << "ImgILibMPEG2BodyC::Decode frameNo=" << allocFrameId <<  " frameType=" << frameTypes[ptype] << "\n");
      uint8_t * buf[3];
      UIntT frameid = allocFrameId++;
      void *id = (void*)frameid;
      //ImageC<ByteRGBValueC> nimg(imgSize[0].V(),imgSize[1].V());
      UIntT imagePixels = imgSize[0].V() * imgSize[1].V();
      SArray1dC<ByteT> buffer(imagePixels * 3);
      images[frameid] = buffer;
      //buf[0] = (uint8_t*) &(nimg[0][0]);
      buf[0] = &(buffer[0]);
      buf[1] = &(buffer[imagePixels]);
      buf[2] = &(buffer[imagePixels + imagePixels/2]);
      mpeg2_set_buf(decoder, buf, id);
#endif
    } break;
      
    case STATE_PICTURE_2ND:
      ONDEBUG(cerr << "ImgILibMPEG2BodyC::Decode got PICTURE_2ND.\n");
      break;
      
    case STATE_END:
      ONDEBUG(cerr << "ImgILibMPEG2BodyC::Decode got END.\n");
      break; 
        
    case STATE_SLICE:    {
      ONDEBUG(cerr << "ImgILibMPEG2BodyC::Decode got SLICE.\n");
      if(info->display_fbuf != 0) {
	UIntT frameid = (UIntT) info->display_fbuf->id;
	IntT frameType = info->display_picture->flags & PIC_MASK_CODING_TYPE;
	ONDEBUG(cerr << "ImgILibMPEG2BodyC::Decode frameid=" << frameid << " " << info->display_picture->temporal_reference << " Type=" << frameTypes[lastFrameType] <<"\n");
	ImageC<ByteRGBValueC> nimg(imgSize[0].V(),imgSize[1].V());
#if 1
	UIntT stride = imgSize[1].V();
	if (info->display_fbuf) {
	  //save_pgm (info->sequence->width, info->sequence->height,
	  //info->display_fbuf->buf, framenum++);
	  ByteT ** xbuf = (ByteT **) info->display_fbuf->buf;
	  UIntT row = 0;
	  for(Array2dIterC<ByteRGBValueC> it(nimg);it;row++) {
	    ByteT *yd = &(xbuf[0][stride * row]); 
	    IntT crow = ((row >> 1) & ~((UIntT)1)) + row % 2;
	      
	    ByteT *ud = &(xbuf[1][(stride/2) * crow]);
	    ByteT *vd = &(xbuf[2][(stride/2) * crow]);
	    do {
	      ByteRGBValueC &p1 = *it;
	      it++;
	      ByteYUV2RGB2(yd[0],yd[1],(*ud + 128),(*vd + 128),p1,*it);
	      yd += 2;
	      ud++;
	      vd++;	      
	    } while(it.Next()) ;
	  }
	}
#endif
	imageCache.Insert(frameNo, Tuple2C<ImageC<ByteRGBValueC>,IntT>(nimg, frameType));
	images.Del(frameid);
	frameNo++;
	gotFrames = true;
      }
    } break;
      
    case STATE_INVALID:
      ONDEBUG(cerr << "ImgILibMPEG2BodyC::Decode got INVALID.\n");
      break;
        
    default:
      cerr << "ImgILibMPEG2BodyC::Decode unknown state=" << m_state << "\n";
      return false;
    }

    return true;
  }

  bool ImgILibMPEG2BodyC::Demultiplex(UIntT &frameNo, const mpeg2_info_t *info, bool &gotFrames)
  {
    ONDEBUG(cerr << "ImgILibMPEG2BodyC::Demultiplex called track=" << m_demuxTrack << "\n");
    
    uint8_t * header;
    int bytes;
    int len;
  
#define NEEDBYTES(x)						                                    \
    do {                                                            \
      int missing;                                                  \
                                                                    \
      missing = (x) - bytes;                                        \
      if (missing > 0) {					                                  \
        if (header == m_headBuf) {				                          \
          if (missing <= bufEnd - bufStart) {			                  \
            memcpy (header + bytes, bufStart, missing);	            \
            bufStart += missing;				                            \
            bytes = (x);				                                    \
          } else {					                                        \
            memcpy (header + bytes, bufStart, bufEnd - bufStart);	  \
            m_demuxStateBytes = bytes + bufEnd - bufStart;		      \
            return 0;					                                      \
          }						                                              \
        } else {						                                        \
          memcpy (m_headBuf, header, bytes);		                    \
          m_demuxState = DEMUX_HEADER;				                      \
          m_demuxStateBytes = bytes;				                        \
          return 0;					                                        \
        }							                                              \
      }							                                                \
    } while (0)
  
#define DONEBYTES(x)		                                            \
    do {			                                                      \
      if (header != m_headBuf)	                                    \
      bufStart = header + (x);	                                    \
    } while (0)
  
    switch (m_demuxState)
    {
      case DEMUX_HEADER:
        if (m_demuxStateBytes > 0)
        {
          header = m_headBuf;
          bytes = m_demuxStateBytes;
          goto continue_header;
        }
        break;
        
      case DEMUX_DATA:
        if (m_demuxStateBytes > bufEnd - bufStart)
        {
//          ONDEBUG(cerr << "ImgILibMPEG2BodyC::Demultiplex (0) start=" << (UIntT)bufStart << ", left=" << (UIntT)(bufEnd - bufStart) << endl;)

          DecodeBlock(frameNo, info, gotFrames, bufStart, bufEnd);

          m_demuxStateBytes -= bufEnd - bufStart;
          return 0;
        }

//        ONDEBUG(cerr << "ImgILibMPEG2BodyC::Demultiplex (1) start=" << (UIntT)bufStart << ", chunk=" << (UIntT)(m_demuxStateBytes) << ", left=" << (UIntT)(bufEnd - bufStart) << endl;)

        DecodeBlock(frameNo, info, gotFrames, bufStart, bufStart + m_demuxStateBytes);

        bufStart += m_demuxStateBytes;
        break;
        
      case DEMUX_SKIP:
        if (m_demuxStateBytes > bufEnd - bufStart)
        {
          m_demuxStateBytes -= bufEnd - bufStart;

          return 0;
        }
        bufStart += m_demuxStateBytes;
        break;
    }
    
    while (true)
    {
    payload_start:
      header = bufStart;
      bytes = bufEnd - bufStart;

    continue_header:
      NEEDBYTES (4);
      if (header[0] || header[1] || (header[2] != 1))
      {
        if (header != m_headBuf)
        {
          bufStart++;
          goto payload_start;
        }
        else
        {
          header[0] = header[1];
          header[1] = header[2];
          header[2] = header[3];
          bytes = 3;
          goto continue_header;
        }
      }
      
      switch (header[3])
      {
        case 0xb9:	/* program end code */
          /* DONEBYTES (4); */
          /* break;         */

          return 1;
          
        case 0xba:	/* pack header */
          NEEDBYTES (5);
          if ((header[4] & 0xc0) == 0x40)
          {	/* mpeg2 */
            NEEDBYTES (14);
            len = 14 + (header[13] & 7);
            NEEDBYTES (len);
            DONEBYTES (len);
            /* header points to the mpeg2 pack header */
          }
          else
            if ((header[4] & 0xf0) == 0x20)
            {	/* mpeg1 */
              NEEDBYTES (12);
              DONEBYTES (12);
              /* header points to the mpeg1 pack header */
            }
            else
            {
              fprintf (stderr, "weird pack header\n");
              DONEBYTES (5);
            }
          break;
          
        default:
          if (header[3] == m_demuxTrack)
          {
            NEEDBYTES (7);
            if ((header[6] & 0xc0) == 0x80)
            {	/* mpeg2 */
              NEEDBYTES (9);
              len = 9 + header[8];
              NEEDBYTES (len);
              /* header points to the mpeg2 pes header */
              if (header[7] & 0x80)
              {
                uint32_t pts;
          
                pts = (((header[9] >> 1) << 30) |
                       (header[10] << 22) | ((header[11] >> 1) << 15) |
                       (header[12] << 7) | (header[13] >> 1));
                mpeg2_pts (decoder, pts);
              }
            }
            else
            {	/* mpeg1 */
              int len_skip;
              uint8_t * ptsbuf;
          
              len = 7;
              while (header[len - 1] == 0xff)
              {
                len++;
                NEEDBYTES (len);
                if (len > 23)
                {
                  fprintf (stderr, "too much stuffing\n");
                  break;
                }
              }
              if ((header[len - 1] & 0xc0) == 0x40)
              {
                len += 2;
                NEEDBYTES (len);
              }
              len_skip = len;
              len += mpeg1_skip_table[header[len - 1] >> 4];
              NEEDBYTES (len);
              /* header points to the mpeg1 pes header */
              ptsbuf = header + len_skip;
              if (ptsbuf[-1] & 0x20)
              {
                uint32_t pts;
                
                pts = (((ptsbuf[-1] >> 1) << 30) |
                       (ptsbuf[0] << 22) | ((ptsbuf[1] >> 1) << 15) |
                       (ptsbuf[2] << 7) | (ptsbuf[3] >> 1));
                mpeg2_pts(decoder, pts);
              }
            }
            DONEBYTES (len);
            bytes = 6 + (header[4] << 8) + header[5] - len;
            if (bytes > bufEnd - bufStart)
            {
//              ONDEBUG(cerr << "ImgILibMPEG2BodyC::Demultiplex (2) start=" << (UIntT)bufStart << ", left=" << (UIntT)(bufEnd - bufStart) << endl;)

              DecodeBlock(frameNo, info, gotFrames, bufStart, bufEnd);
              
              m_demuxState = DEMUX_DATA;
              m_demuxStateBytes = bytes - (bufEnd - bufStart);
              return 0;
            }
            else
              if (bytes > 0)
              {
//                ONDEBUG(cerr << "ImgILibMPEG2BodyC::Demultiplex (3) start=" << (UIntT)bufStart << ", chunk= " << (UIntT)(bytes) << ", left=" << (UIntT)(bufEnd - bufStart) << endl;)

                DecodeBlock(frameNo, info, gotFrames, bufStart, bufStart + bytes);

                bufStart += bytes;
              }
            }
            else
              if (header[3] < 0xb9)
              {
                fprintf (stderr,
                "looks like a video stream, not system stream\n");
                DONEBYTES (4);
              }
              else
              {
                NEEDBYTES (6);
                DONEBYTES (6);
                bytes = (header[4] << 8) + header[5];
                if (bytes > bufEnd - bufStart)
                {
                  m_demuxState = DEMUX_SKIP;
                  m_demuxStateBytes = bytes - (bufEnd - bufStart);

                  return 0;
                }
                bufStart += bytes;
              }
      }
    }
  }
  
  bool ImgILibMPEG2BodyC::DecodeBlock(UIntT &frameNo, const mpeg2_info_t *info, bool &gotFrames, ByteT *start, ByteT *end)
  {
#if 0
    ofstream file_out("test.mpg", ios::out|ios::binary|ios::app);
    file_out.write(reinterpret_cast<char*>(start), end - start);
    file_out.close();
#endif

    // Feed in the new data
    mpeg2_buffer(decoder, start, end);

    // Keep decoding until we've emptied the decoder
    while ((m_state = mpeg2_parse(decoder)) != -1)
    {
      Decode(frameNo, info, gotFrames);
    }
    
    return true;
  }
  
  //: Get next frame.
  
  bool ImgILibMPEG2BodyC::Get(ImageC<ByteRGBValueC> &img)
  {
    ONDEBUG(cerr << "ImgILibMPEG2BodyC::Get called. \n");
    Tuple2C<ImageC<ByteRGBValueC>,IntT> dat;
    if (!imageCache.LookupR(frameNo, dat))
    {
      if (m_demuxTrack >= 0)
      {
        if (!DemultiplexGOP(frameNo))
        {
          cerr << "ImgILibMPEG2BodyC::Get failed to demultiplex GOP.\n";
          return false;
        }
      }
      else
        if (!DecodeGOP(frameNo))
        {
          cerr << "ImgILibMPEG2BodyC::Get failed to decode GOP.\n";
          return false;
        }
    }
    if (!imageCache.LookupR(frameNo, dat))
    {
      // Probably a seek forward in the file to a GOP we haven't seen yet...
      return false;
    }
    
    img = dat.Data1();
    lastFrameType = dat.Data2();
    frameNo++;   
    return true;
  }
  
  //: Seek to location in stream.
  
  bool ImgILibMPEG2BodyC::Seek(UIntT off) {
    ONDEBUG(cerr << "ImgILibMPEG2BodyC::Seek called dest=" << off << "\n");
    // Move the decoder to the next gop before changing the buffer.
    frameNo = off;
    return true;
  }

  //: Seek to location in stream.
  
  bool ImgILibMPEG2BodyC::Seek64(StreamPosT off) {
    ONDEBUG(cerr << "ImgILibMPEG2BodyC::Seek called dest=" << off << "\n");
    frameNo = off;
    return true;
  }
  
  //: Build GOP index to frame.
  
  bool ImgILibMPEG2BodyC::BuildIndex(UIntT targetFrame)
  {
    /*
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
    */
    
    return true;
  }
  
  //: Skip to next start code.
  
  /*
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
  */

}

