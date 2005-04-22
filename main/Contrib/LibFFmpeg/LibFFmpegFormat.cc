// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU
// General Public License (GPL). See the gpl.licence file for details or
// see http://www.gnu.org/copyleft/gpl.html
// file-header-ends-here
//////////////////////////////////////////////////////////////////
//! rcsid = "$Id$"
//! lib = RavlLibFFmpeg
//! author = "Warren Moore"
//! file="Ravl/Contrib/LibFFmpeg/LibFFmpegFormat.cc"

#include "Ravl/Image/LibFFmpegFormat.hh"
#include "Ravl/Image/ImgIOFFmpeg.hh"
#include <ctype.h>
#include "Ravl/DP/ByteFileIO.hh"
#include "Ravl/DP/SPortAttach.hh"

#include <ffmpeg/avcodec.h>
#include <ffmpeg/avformat.h>

#define DODEBUG 1

#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlImageN
{



  void InitLibFFmpegFormat()
  {}



  FileFormatLibFFmpegBodyC::FileFormatLibFFmpegBodyC() :
    FileFormatBodyC("ffmpeg", "FFmpeg file input.")
  {
    // Register all formats and codecs
    av_register_all();
  }



  const type_info &FileFormatLibFFmpegBodyC::ProbeLoad(IStreamC &in, const type_info &obj_type) const
  {
    ONDEBUG(cerr << "FileFormatLibFFmpegBodyC::ProbeLoad(IStreamC &,...) called" << endl);
    
    if (!in.good())
      return typeid(void);
    
    ONDEBUG(cerr << "FileFormatLibFFmpegBodyC::ProbeLoad(IStreamC&,...) not an FFmpeg supported file" << endl);
    return typeid(void); 
  }



  const type_info &FileFormatLibFFmpegBodyC::ProbeLoad(const StringC &filename,
                                                      IStreamC &in,
                                                      const type_info &obj_type) const
  {
    ONDEBUG(cerr << "FileFormatLibFFmpegBodyC::ProbeLoad(const StringC&,IStreamC&,...) called (" << filename << ")" << endl);
    
    if (!in.good())
      return typeid(void);
    
    if (IsSupported(filename.chars()))
    {
      return typeid(ImageC<ByteRGBValueC>);
    }
    
    ONDEBUG(cerr << "FileFormatLibFFmpegBodyC::ProbeLoad(const StringC&,IStreamC&,...) not an FFmpeg supported file (" << filename << ")" << endl);
    return typeid(void);
  }



  const type_info &FileFormatLibFFmpegBodyC::ProbeSave(const StringC &filename,
                                                      const type_info &obj_type,
                                                      bool forceFormat ) const
  {
    ONDEBUG(cerr << "FileFormatLibFFmpegBodyC::ProbeSave(const StringC&,...) not supported" << endl);
    return typeid(void);   
  }



  DPIPortBaseC FileFormatLibFFmpegBodyC::CreateInput(const StringC &filename, const type_info &obj_type) const
  {
    ONDEBUG(cerr << "FileFormatLibFFmpegBodyC::CreateInput(const StringC&,...) called (" << filename << ")" << endl);
    
    if (IsSupported(filename.chars()))
    {
      //return SPort(DPIByteFileC(fn) >> ImgILibFFmpegC(true));
      return DPIPortBaseC();
    }
    
    ONDEBUG(cerr << "FileFormatLibFFmpegBodyC::CreateInput(const StringC&,...) not an FFmpeg supported file (" << filename << ")" << endl);
    return DPIPortBaseC();
  }



  DPOPortBaseC FileFormatLibFFmpegBodyC::CreateOutput(const StringC &filename, const type_info &obj_type) const
  {
    return DPOPortBaseC();  
  }



  DPIPortBaseC FileFormatLibFFmpegBodyC::CreateInput(IStreamC &in, const type_info &obj_type) const
  {
    return DPIPortBaseC();
  }



  DPOPortBaseC FileFormatLibFFmpegBodyC::CreateOutput(OStreamC &out, const type_info &obj_type) const
  {
    return DPOPortBaseC();  
  }



  const type_info &FileFormatLibFFmpegBodyC::DefaultType() const
  { 
    return typeid(ImageC<ByteRGBValueC>); 
  }



  bool FileFormatLibFFmpegBodyC::IsSupported(const char *filename) const
  {
    ONDEBUG(cerr << "FileFormatLibFFmpegBodyC::IsSupported(const char *) called (" << filename << ")" << endl);
    
    // Open video file
    bool ret = false;
    AVFormatContext *pFormatCtx;
    if (av_open_input_file(&pFormatCtx, filename, NULL, 0, NULL) == 0)
    {
      // Retrieve stream information
      if (av_find_stream_info(pFormatCtx) >= 0)
      {
        ONDEBUG(dump_format(pFormatCtx, 0, filename, false));
        
        // Find the first video stream
        IntT videoStream = -1;
        for (IntT i = 0; i < pFormatCtx->nb_streams; i++)
        {
          if (pFormatCtx->streams[i]->codec.codec_type == CODEC_TYPE_VIDEO)
          {
            videoStream = i;
            break;
          }
        }
        
        if (videoStream != -1)
        {
          ONDEBUG(cerr << "FileFormatLibFFmpegBodyC::IsSupported stream(" << videoStream << ")" << endl);
          
          // Get a pointer to the codec context for the video stream
          AVCodecContext *pCodecCtx = &pFormatCtx->streams[videoStream]->codec;
      
          // Find the decoder for the video stream
          AVCodec *pCodec = avcodec_find_decoder(pCodecCtx->codec_id);
          if (pCodec != NULL)
          {
            ONDEBUG(cerr << "FileFormatLibFFmpegBodyC::IsSupported codec found(" << (pCodec->name != NULL ? pCodec->name : "NULL") << ")" << endl);
            
            // Inform the codec that we can handle truncated bitstreams
            // i.e. bitstreams where frame boundaries can fall in the middle of packets
            if (pCodec->capabilities & CODEC_CAP_TRUNCATED)
                pCodecCtx->flags |= CODEC_FLAG_TRUNCATED;
        
            // Open codec
            if (avcodec_open(pCodecCtx, pCodec) >= 0)
            {
              ret = true;
            }
            
            // Clean up codec
            avcodec_close(pCodecCtx);
          }
        }
      }
      
      // Close the video file
      av_close_input_file(pFormatCtx);    
    }

    ONDEBUG(cerr << "FileFormatLibFFmpegBodyC::IsSupported(const char *) " << (ret ? "succeeded" : "failed") << endl);
    return ret;
  }
  
  
  
  static FileFormatLibFFmpegC Init;  



}
