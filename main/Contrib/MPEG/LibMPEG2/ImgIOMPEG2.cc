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
#include "Ravl/DP/AttributeValueTypes.hh"

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
  
  static const char frameTypes[5] = { 'X', 'I', 'P', 'B', 'D' };
  
  static const int mpeg1_skip_table[16] = { 0, 0, 4, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

  static const IntT g_cacheStep = 20;
  
  //: Default constructor.
  
  ImgILibMPEG2BodyC::ImgILibMPEG2BodyC(IntT demuxTrack) :
    m_decoder(0),
    m_state(-2),
    m_buffer(16384 * 2),
    m_bufStart(0),
    m_bufEnd(0),
    m_allocFrameId(0),
    m_frameNo(0),
    m_maxFrameIndex(0),
    m_lastRead(0),
    m_offsets(true),
    m_imageCache(g_cacheStep * 2),
    m_sequenceInit(false),
    m_lastFrameType(0),
    m_demuxTrack(demuxTrack),
    m_demuxState(DEMUX_SKIP),
    m_demuxStateBytes(0),
    m_gopCount(0),
    m_gopLimit(0),
    m_previousGop(0),
    m_blockRead(0)
  {
    m_decoder = mpeg2_init();
    
    // Check the demux track
    if (m_demuxTrack > 0 && (m_demuxTrack < 0xe0 || m_demuxTrack > 0xef))
    {
      cerr << "ImgILibMPEG2BodyC::ImgILibMPEG2BodyC invalid demultiplex track." << endl;
      m_demuxTrack = -1;
    }

    BuildAttributes();
  }
  
  //: Destructor.
  
  ImgILibMPEG2BodyC::~ImgILibMPEG2BodyC()
  {
    if(m_decoder != 0)
      mpeg2_close(m_decoder);
  }
  
  bool ImgILibMPEG2BodyC::InitialSeek()
  {
    RavlAssertMsg(m_offsets.IsEmpty(), "ImgILibMPEG2BodyC::InitialSeek called with non-empty offsets");

    if (input.IsValid())
    {
      DPSeekCtrlC seek(input);
      if (seek.IsValid())
      {
        m_offsets.Insert(0, seek.Tell64());
        
        return true;
      }
    }
    
    return false;
  }

  //: Read data from stream into buffer.
  
  bool ImgILibMPEG2BodyC::ReadData()
  {
    RavlAssertMsg(input.IsValid(), "ImgILibMPEG2BodyC::ReadData called with invalid input");
    
    DPSeekCtrlC seek(input);
    if (!seek.IsValid())
      return false;
    
    if (input.IsGetEOS())
      return false;

    m_bufStart = &(m_buffer[0]);
    m_lastRead = seek.Tell64();
    UIntT len = input.GetArray(m_buffer);
    
    m_bufEnd = m_bufStart + len;
    return (len > 0);
  }

  //: Register stream attributes.

  void ImgILibMPEG2BodyC::BuildAttributes() {
    RegisterAttribute(AttributeTypeStringC("frametype","MPEG frame type",true,false));
  }


  // Returns false if the attribute name is unknown.
  // This is for handling stream attributes such as frame rate, and compression ratios.
  
  bool ImgILibMPEG2BodyC::GetAttr(const StringC &attrName,StringC &attrValue)
  {
    if(attrName == "frametype") {
      attrValue = StringC(frameTypes[m_lastFrameType]);
      return true; 
    }
    return DPPortBodyC::GetAttr(attrName,attrValue);
  }
  

  //: Decode a whole GOP and put it in the image cache.
  
  bool ImgILibMPEG2BodyC::DecodeGOP(UIntT firstFrameNo)
  {
//    ONDEBUG(cerr << "ImgILibMPEG2BodyC::DecodeGOP called last state (" << m_state << ")" << endl;)
    
    // If not set, get the initial seek position
    if (m_offsets.IsEmpty() && !InitialSeek())
    {
      cerr << "ImgILibMPEG2BodyC::DecodeGOP unable to find initial seek position" << endl;
      return false;
    }
    
    // Find the GOP before the required frame...
    StreamPosT at;
    while(!m_offsets.Find(firstFrameNo, at))
    {
      if(firstFrameNo == 0)
        return false;
      firstFrameNo--;
    }
    ONDEBUG(cerr << "ImgILibMPEG2BodyC::DecodeGOP seeking frame (" << firstFrameNo << ") at (" << at << ")" << endl;)
    
    // Decoder vars
    UIntT localFrameNo = firstFrameNo;
    m_gopLimit = 1;

    // Find the gop before the required one
    UIntT previousFrameNo = firstFrameNo;
    if (firstFrameNo > 0)
    {
      previousFrameNo = firstFrameNo - 1;
      StreamPosT prevAt;
      while(!m_offsets.Find(previousFrameNo, prevAt))
      {
        if(previousFrameNo == 0)
          return false;
        previousFrameNo--;
      }
      
      // Only step back another GOP if we're didn't  read it last GOP
      ONDEBUG(cerr << "ImgILibMPEG2BodyC::DecodeGOP previous gop frame (" << m_previousGop << ")" << endl;)
      if (m_previousGop != previousFrameNo)
      {
        // Get the previous GOP info
        at = prevAt;
        localFrameNo = previousFrameNo;
        
        if (previousFrameNo == 0)
          // Skip the GOP at the beginning of the sequence, flush the first GOP through the decoder then store the next
          m_gopLimit = 3;
        else
          // Flush the first GOP through the decoder then store the next
          m_gopLimit = 2;
        ONDEBUG(cerr << "ImgILibMPEG2BodyC::DecodeGOP seeking previous frame (" << previousFrameNo << ") at (" << at << ")" << endl;)
      }
    }
    else
      // Skip over the GOP at the beginning of the sequence
      m_gopLimit = 2;
    
    // Seek to the required position
    DPSeekCtrlC seek(input);
    if (!seek.IsValid())
      return false;
    seek.Seek64(at);
    
    // Reset the decoder if the first frame of the GOP is zero.
    if (localFrameNo == 0)
    {
      // Close the decoder
      mpeg2_close(m_decoder);
      
      // Restart the decoder
      m_decoder = mpeg2_init();
      m_sequenceInit = false;
    }

    // Get the MPEG info structure
    const mpeg2_info_t *info = mpeg2_info(m_decoder);

    // Set the state vars
    m_state = -1;
    m_gopCount = 0;
    m_endFound = false;
    
    // Decoder loop
    do
    {
      // Read some data into the decoder if necessary
      if (m_state == -1)
      {
        if(ReadData())
          mpeg2_buffer(m_decoder, m_bufStart, m_bufEnd);
        else
        {
          ONDEBUG(cerr << "ImgILibMPEG2BodyC::DecodeGOP failed to read data." << endl;)
          m_state = -2;
          return false;
        }
      }

      // Decode the buffer
      m_state = mpeg2_parse(m_decoder);
      Decode(localFrameNo, info);

      // Check we've decoded a complete GOP
      if (m_gopCount >= m_gopLimit)
      {
        // Store the gop frame number
        m_previousGop = firstFrameNo;

        // Store the read pos
        if (!m_endFound)
        {
          UIntT parsePos = m_lastRead + (UIntT) (m_decoder->buf_start - &(m_buffer[0]));
          ONDEBUG(cerr << "ImgILibMPEG2BodyC::DecodeGOP **** Recording frame (" << localFrameNo << ") at (" << parsePos << ")" << endl;)
          m_offsets.Insert(localFrameNo, parsePos, false);
        }

        // Update the max frame
        if(localFrameNo > m_maxFrameIndex)
          m_maxFrameIndex = localFrameNo;
        
        break;
      }
    } while(true);

    return true;
  }
  
  
  bool ImgILibMPEG2BodyC::DemultiplexGOP(UIntT firstFrameNo)
  {
    // If not set, get the initial seek position
    if (m_offsets.IsEmpty() && !InitialSeek())
    {
      cerr << "ImgILibMPEG2BodyC::DemultiplexGOP unable to find initial seek position" << endl;
      return false;
    }
    
    // Find the GOP before the required frame...
    StreamPosT at;
    while(!m_offsets.Find(firstFrameNo, at) || at == -1)
    {
      if(firstFrameNo == 0)
        return false;
      firstFrameNo--;
    }
    ONDEBUG(cerr << "ImgILibMPEG2BodyC::DemultiplexGOP seeking frame (" << firstFrameNo << ") at (" << at << ")" << endl;)
    
    // Decoder vars
    UIntT localFrameNo = firstFrameNo;
    m_gopLimit = 1;

    // Find the gop before the required one
    UIntT previousFrameNo = firstFrameNo;
    if (firstFrameNo > 0)
    {
      previousFrameNo = firstFrameNo - 1;
      StreamPosT prevAt;
      while(!m_offsets.Find(previousFrameNo, prevAt) || prevAt == -1)
      {
        if(previousFrameNo == 0)
          return false;
        previousFrameNo--;
      }
      
      // Only step back another GOP if we're didn't read it last GOP
      ONDEBUG(cerr << "ImgILibMPEG2BodyC::DemultiplexGOP previous gop frame (" << m_previousGop << ")" << endl;)
      if (m_previousGop != previousFrameNo)
      {
        // Get the previous GOP info
        ONDEBUG(cerr << "ImgILibMPEG2BodyC::DemultiplexGOP seeking previous frame (" << previousFrameNo << ") at (" << at << ")" << endl;)
        at = prevAt;
        localFrameNo = previousFrameNo;
        
        // Restart the demultixer
        m_demuxState = DEMUX_SKIP;
        m_demuxStateBytes = 0;
        
        // All existing data in the decoder is invalid
        m_blockRead = 0;
        
        // Skip over the first GOP, flush the next through the decoder then store the next
        m_gopLimit = 3;
      }
    }
    else
      // Skip over the GOP at the beginning of the sequence
      m_gopLimit = 2;
    
    // Seek to the required position
    DPSeekCtrlC seek(input);
    if (!seek.IsValid())
      return false;
    seek.Seek64(at);
    
    // Reset the decoder if the first frame of the GOP is zero.
    if (localFrameNo == 0)
    {
      // Close the decoder
      mpeg2_close(m_decoder);
      
      // Restart the decoder
      m_decoder = mpeg2_init();
      m_sequenceInit = false;
      
      // Restart the demultiplexer
      m_demuxState = DEMUX_SKIP;
      m_demuxStateBytes = 0;
      
      // All existing data in the decoder is invalid
      m_blockRead = 0;
    }

    // Get the MPEG info structure
    const mpeg2_info_t *info = mpeg2_info(m_decoder);

    // Set the state vars
    m_state = -1;
    m_gopCount = 0;
    m_gopDone = false;
    m_endFound = false;

    // Decoder loop
    do
    {
      // Read some data into the decoder if necessary
      if(!ReadData())
      {
        ONDEBUG(cerr << "ImgILibMPEG2BodyC::DemultiplexGOP failed to read data." << endl;)
        m_state = -2;
        return false;
      }

      // Decode the buffer
      Demultiplex(localFrameNo, info, firstFrameNo);

      // Stop at the end
      if (m_endFound)
        return false;

      // Check we've decoded all required GOPs
      if (m_gopCount >= m_gopLimit)
      {
        ONDEBUG(cerr << "ImgILibMPEG2BodyC::DemultiplexGOP read (" << m_gopCount << ") GOPs" << endl;)
        break;
      }
    } while(true);

    return true;
  }
  
  bool ImgILibMPEG2BodyC::Decode(UIntT &frameNo, const mpeg2_info_t *info)
  {
//    ONDEBUG(cerr << "ImgILibMPEG2BodyC::Decode state (" << m_state << ")" << endl;)

    switch(m_state)
    {
    case -1:   // Got to end of buffer ?
//      ONDEBUG(cerr << "ImgILibMPEG2BodyC::Decode need input" << endl;)
      break;
        
    case STATE_SEQUENCE_REPEATED:
    case STATE_SEQUENCE:{ 
      ONDEBUG(cerr << "ImgILibMPEG2BodyC::Decode got SEQUENCE." << endl;)
      if(!m_sequenceInit) {
        m_sequenceInit = true;
        m_imgSize = Index2dC(info->sequence->height,info->sequence->width);
#if USE_OWN_RGBCONV
	      //mpeg2_convert(m_decoder, my_convert_rgb , NULL);
#else
	      //mpeg2_convert(m_decoder, convert_rgb24, NULL);
#endif
	      //mpeg2_custom_fbuf(m_decoder, 1);
        info = mpeg2_info(m_decoder);
        ONDEBUG(cerr << "ImgILibMPEG2BodyC::Decode imageSize (" << m_imgSize << ")" << endl;)
      }
    } break;
      
    case STATE_GOP:
      ONDEBUG(cerr << "ImgILibMPEG2BodyC::Decode got STATE_GOP (" << m_gopCount << ") frame (" << frameNo << ")" << endl;)
      m_gopCount++;
      
      // Check it's not a skipped GOP
      {
        StreamPosT at;
        if (m_offsets.Find(frameNo, at) && at == -1)
        {
          ONDEBUG(cerr << "ImgILibMPEG2BodyC::Decode skipped GOP encountered" << endl;)
          m_gopLimit++;
        }
      }
      break;
	
    case STATE_PICTURE: {
#if 0	
      ONDEBUG(cerr << "ImgILibMPEG2BodyC::Decode got PICTURE." << endl;)
      ONDEBUG(UIntT ptype = info->current_picture->flags & PIC_MASK_CODING_TYPE;)
      ONDEBUG(cerr << "ImgILibMPEG2BodyC::Decode frameNo(" << m_allocFrameId << ") frameType(" << frameTypes[ptype] << ")" << endl;)
      uint8_t * buf[3];
      UIntT frameid = m_allocFrameId++;
      void *id = (void*)frameid;
      //ImageC<ByteRGBValueC> nimg(m_imgSize[0].V(), m_imgSize[1].V());
      UIntT imagePixels = m_imgSize[0].V() * m_imgSize[1].V();
      SArray1dC<ByteT> buffer(imagePixels * 3);
      m_images[frameid] = buffer;
      //buf[0] = (uint8_t*) &(nimg[0][0]);
      buf[0] = &(buffer[0]);
      buf[1] = &(buffer[imagePixels]);
      buf[2] = &(buffer[imagePixels + imagePixels/2]);
      mpeg2_set_buf(m_decoder, buf, id);
#endif
    } break;
      
    case STATE_PICTURE_2ND:
      ONDEBUG(cerr << "ImgILibMPEG2BodyC::Decode got PICTURE_2ND." << endl;)
      break;
      
    case STATE_END:
      ONDEBUG(cerr << "ImgILibMPEG2BodyC::Decode got END." << endl;)
      m_endFound = true;
      break; 
        
    case STATE_SLICE: {
      ONDEBUG(cerr << "ImgILibMPEG2BodyC::Decode got SLICE." << endl;)
      // Ignore the slice if it's well before what we need, or no image is ready
      if ((m_gopLimit - m_gopCount < 3) && info->display_fbuf != 0)
      {
        UIntT frameid = (UIntT) info->display_fbuf->id;
        
        // Only store the image if it's fresh
        if (m_gopLimit - m_gopCount <= 1)
        {
          IntT frameType = info->display_picture->flags & PIC_MASK_CODING_TYPE;
          ONDEBUG(cerr << "ImgILibMPEG2BodyC::Decode frameid(" << frameid << ") temporal ref(" << info->display_picture->temporal_reference << ") type(" << frameTypes[frameType] << ")" << endl;)
          ImageC<ByteRGBValueC> nimg(m_imgSize[0].V(), m_imgSize[1].V());
#if 1
          UIntT stride = m_imgSize[1].V();
          if (info->display_fbuf)
          {
            //save_pgm (info->sequence->width, info->sequence->height,
            //info->display_fbuf->buf, framenum++);
            ByteT ** xbuf = (ByteT **) info->display_fbuf->buf;
            UIntT row = 0;
            for(Array2dIterC<ByteRGBValueC> it(nimg);it;row++)
            {
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
          // Cache the image
          ONDEBUG(cerr << "ImgILibMPEG2BodyC::Decode caching frame (" << frameNo << ")" << endl;)
          m_imageCache.Insert(frameNo, Tuple2C<ImageC<ByteRGBValueC>,IntT>(nimg, frameType));
          m_images.Del(frameid);
        }
        else
        {
          // Flush the image through the decoder
          ONDEBUG(cerr << "ImgILibMPEG2BodyC::Decode skipping frame (" << frameNo << ")" << endl;)
        }
        
        // Can cause problems if a slice is found after a GOP, so skip this one
        if (m_gopLimit - m_gopCount < 1)
        {
          ONDEBUG(cerr << "ImgILibMPEG2BodyC::Decode **** replacing GOP as frame (" << frameNo << ") found after" << endl;)
          m_offsets.Insert(frameNo, -1, true);
          m_gopLimit++;
          m_gopDone = false;
        }

        // Update the frame count
        frameNo++;
        
        // Check the cache size
        UIntT cacheSize = (m_gopLimit - 1) * g_cacheStep;
        if (cacheSize > m_imageCache.MaxSize())
        {
          ONDEBUG(cerr << "ImgILibMPEG2BodyC::Decode resized image cache to (" << cacheSize << ")" << endl;)
          m_imageCache.MaxSize(cacheSize);
        }
      }
    } break;
      
    case STATE_INVALID:
      cerr << "ImgILibMPEG2BodyC::Decode got INVALID." << endl;
      break;
        
    default:
      cerr << "ImgILibMPEG2BodyC::Decode unknown state=" << m_state << endl;
      return false;
    }

    return true;
  }

  bool ImgILibMPEG2BodyC::Demultiplex(UIntT &frameNo, const mpeg2_info_t *info, const UIntT firstFrameNo)
  {
//    ONDEBUG(cerr << "ImgILibMPEG2BodyC::Demultiplex called track (" << m_demuxTrack << ")" << endl;)
    
    uint8_t * header;
    int bytes;
    int len;
  
#define NEEDBYTES(x)                                                    \
    do {                                                                \
      int missing;                                                      \
                                                                        \
      missing = (x) - bytes;                                            \
      if (missing > 0) {                                                \
        if (header == m_headBuf) {                                      \
          if (missing <= m_bufEnd - m_bufStart) {                       \
            memcpy(header + bytes, m_bufStart, missing);                \
            m_bufStart += missing;                                      \
            bytes = (x);				                                        \
          } else {					                                            \
            memcpy(header + bytes, m_bufStart, m_bufEnd - m_bufStart);  \
            m_demuxStateBytes = bytes + m_bufEnd - m_bufStart;          \
            return false;                                               \
          }                                                             \
        } else {						                                            \
          memcpy(m_headBuf, header, bytes);		                          \
          m_demuxState = DEMUX_HEADER;                                  \
          m_demuxStateBytes = bytes;                                    \
          return false;                                                 \
        }                                                               \
      }                                                                 \
    } while (false)
  
#define DONEBYTES(x)                                                    \
    do {                                                                \
      if (header != m_headBuf)                                          \
        m_bufStart = header + (x);                                      \
    } while (false)
  
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
        if (m_demuxStateBytes > m_bufEnd - m_bufStart)
        {
          DecodeBlock(frameNo, info, firstFrameNo, m_bufStart, m_bufEnd, -1);

          m_demuxStateBytes -= m_bufEnd - m_bufStart;
          return 0;
        }

        DecodeBlock(frameNo, info, firstFrameNo, m_bufStart, m_bufStart + m_demuxStateBytes, -1);

        m_bufStart += m_demuxStateBytes;
        break;
        
      case DEMUX_SKIP:
        if (m_demuxStateBytes > m_bufEnd - m_bufStart)
        {
          m_demuxStateBytes -= m_bufEnd - m_bufStart;

          return 0;
        }
        m_bufStart += m_demuxStateBytes;
        break;
    }
    
    while (true)
    {
    payload_start:
      header = m_bufStart;
      bytes = m_bufEnd - m_bufStart;

    continue_header:
      NEEDBYTES(4);
      if (header[0] || header[1] || (header[2] != 1))
      {
        if (header != m_headBuf)
        {
          m_bufStart++;
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
      StreamPosT parsePos = m_lastRead + (m_bufStart - &(m_buffer[0]));
      
      switch (header[3])
      {
        case 0xb9:	/* program end code */
          /* DONEBYTES (4); */
          /* break;         */

          return 1;
          
        case 0xba:	/* pack header */
          NEEDBYTES(5);
          if ((header[4] & 0xc0) == 0x40)
          {	/* mpeg2 */
            NEEDBYTES(14);
            len = 14 + (header[13] & 7);
            NEEDBYTES(len);
            DONEBYTES(len);
            /* header points to the mpeg2 pack header */
          }
          else
          {
            if ((header[4] & 0xf0) == 0x20)
            {	/* mpeg1 */
              NEEDBYTES(12);
              DONEBYTES(12);
              /* header points to the mpeg1 pack header */
            }
            else
            {
              cerr << "ImgILibMPEG2BodyC::Demultiplex weird pack header" << endl;
              DONEBYTES(5);
            }
          }
          break;
          
        default:
          if (header[3] == m_demuxTrack)
          {
            NEEDBYTES(7);
            if ((header[6] & 0xc0) == 0x80)
            {	/* mpeg2 */
              NEEDBYTES(9);
              len = 9 + header[8];
              NEEDBYTES(len);
              /* header points to the mpeg2 pes header */
              if (header[7] & 0x80)
              {
                uint32_t pts;
          
                pts = (((header[9] >> 1) << 30) |
                       (header[10] << 22) | ((header[11] >> 1) << 15) |
                       (header[12] << 7) | (header[13] >> 1));
                mpeg2_pts(m_decoder, pts);
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
                NEEDBYTES(len);
                if (len > 23)
                {
                  cerr << "ImgILibMPEG2BodyC::Demultiplex too much stuffing" << endl;
                  break;
                }
              }
              if ((header[len - 1] & 0xc0) == 0x40)
              {
                len += 2;
                NEEDBYTES(len);
              }
              len_skip = len;
              len += mpeg1_skip_table[header[len - 1] >> 4];
              NEEDBYTES(len);
              /* header points to the mpeg1 pes header */
              ptsbuf = header + len_skip;
              if (ptsbuf[-1] & 0x20)
              {
                uint32_t pts;
                
                pts = (((ptsbuf[-1] >> 1) << 30) |
                       (ptsbuf[0] << 22) | ((ptsbuf[1] >> 1) << 15) |
                       (ptsbuf[2] << 7) | (ptsbuf[3] >> 1));
                mpeg2_pts(m_decoder, pts);
              }
            }
            DONEBYTES(len);
            bytes = 6 + (header[4] << 8) + header[5] - len;
            if (bytes > m_bufEnd - m_bufStart)
            {
              DecodeBlock(frameNo, info, firstFrameNo, m_bufStart, m_bufEnd, parsePos);
              
              m_demuxState = DEMUX_DATA;
              m_demuxStateBytes = bytes - (m_bufEnd - m_bufStart);
              return 0;
            }
            else
            {
              if (bytes > 0)
              {
                DecodeBlock(frameNo, info, firstFrameNo, m_bufStart, m_bufStart + bytes, parsePos);

                m_bufStart += bytes;
              }
            }
          }
          else
          {
            if (header[3] < 0xb9)
            {
              cerr << "ImgILibMPEG2BodyC::Demultiplex looks like a video stream, not system stream" << endl;
              DONEBYTES(4);
            }
            else
            {
              NEEDBYTES(6);
              DONEBYTES(6);
              bytes = (header[4] << 8) + header[5];
              if (bytes > m_bufEnd - m_bufStart)
              {
                m_demuxState = DEMUX_SKIP;
                m_demuxStateBytes = bytes - (m_bufEnd - m_bufStart);

                return 0;
              }
              m_bufStart += bytes;
            }
          }
          break;
      }
    }
  }
  
  bool ImgILibMPEG2BodyC::DecodeBlock(UIntT &frameNo, const mpeg2_info_t *info, const UIntT firstFrameNo, ByteT *start, ByteT *end, const StreamPosT parsePos)
  {
    // Feed in the new data
    StreamPosT blockPos = m_lastRead + (m_bufStart - &(m_buffer[0]));
    if (blockPos > m_blockRead)
    {
      // Need all of this data
//      ONDEBUG(cerr << "ImgILibMPEG2BodyC::DecodeBlock start(" << blockPos << ") bytes(" << (end - start) << ") read(" << m_blockRead << ")" << endl;)
      mpeg2_buffer(m_decoder, start, end);
    }
    else
    {
      if (m_blockRead < blockPos + (end - start))
      {
        // Some of this data has already been passed to the decoder
//        ONDEBUG(cerr << "ImgILibMPEG2BodyC::DecodeBlock partial start(" << blockPos << ") bytes(" << (end - start) << ") read(" << m_blockRead << ")" << endl;)
        mpeg2_buffer(m_decoder, start + (m_blockRead - blockPos), end);
      }
      else
      {
//        ONDEBUG(cerr << "ImgILibMPEG2BodyC::DecodeBlock skipping start(" << start + (m_blockRead - blockPos) << ") bytes(" << (end - (start + (m_blockRead - blockPos))) << ") read(" << m_blockRead << ")" << endl;)
      }
    }
    m_blockRead = blockPos + (end - start);

    // Keep decoding until we've emptied the decoder
    while ((m_state = mpeg2_parse(m_decoder)) != -1)
    {
      Decode(frameNo, info);
      
      // Check we've decoded a complete GOP
      if (!m_gopDone && m_gopCount == m_gopLimit)
      {
        // Store the gop frame number
        m_previousGop = firstFrameNo;

        // Store the read pos
        if (parsePos >= 0 && !m_endFound)
        {
          ONDEBUG(cerr << "ImgILibMPEG2BodyC::DecodeBlock **** Recording GOP (" << m_gopCount << ") frame (" << frameNo << ") at (" << parsePos << ")" << endl;)
          m_offsets.Insert(frameNo, parsePos, false);
        }

        // Update the max frame
        if(frameNo > m_maxFrameIndex)
          m_maxFrameIndex = frameNo;
        
        // Don't keep caching the info for the rest of the blocks in this demux'd packet
        m_gopDone = true;
      }
    }
    
    return true;
  }
  
  //: Get next frame.
  
  bool ImgILibMPEG2BodyC::Get(ImageC<ByteRGBValueC> &img)
  {
    ONDEBUG(cerr << "ImgILibMPEG2BodyC::Get (" << m_frameNo << ")" << endl;)
    
    bool ok = true;
    Tuple2C<ImageC<ByteRGBValueC>,IntT> dat;
    while (!m_imageCache.LookupR(m_frameNo, dat))
    {
      // If the last read failed, don't bother again
      if (!ok)
      {
        cerr << "ImgILibMPEG2BodyC::Get failed to find frame (" << m_frameNo << ")" << endl;
        return false;
      }
      
      // Keep on truckin' til we find the frame
      if (m_demuxTrack >= 0)
        ok = DemultiplexGOP(m_frameNo);
      else
        ok = DecodeGOP(m_frameNo);
    }
    
    img = dat.Data1();
    m_lastFrameType = dat.Data2();
    m_frameNo++;   
    return true;
  }
  
  //: Seek to location in stream.
  
  bool ImgILibMPEG2BodyC::Seek(UIntT off) {
    ONDEBUG(cerr << "ImgILibMPEG2BodyC::Seek (" << off << ")" << endl;)
    // Move the decoder to the next gop before changing the buffer.
    m_frameNo = off;
    return true;
  }

  //: Seek to location in stream.
  
  bool ImgILibMPEG2BodyC::Seek64(StreamPosT off) {
    ONDEBUG(cerr << "ImgILibMPEG2BodyC::Seek (" << off << ")" << endl;)
    m_frameNo = off;
    return true;
  }
  
}

