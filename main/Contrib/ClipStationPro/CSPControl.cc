// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
/////////////////////////////////////////////////////////////////
//! rcsid="$Id$"
//! lib=CSPDriver
//! file="Contrib/ClipStationPro/CSPControl.cc"

#include "Ravl/Image/CSPControl.hh"
#include "Ravl/OS/DMABuffer.hh"
#include <string.h>

#define DODEBUG 1
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlImageN {

  ClipStationProDeviceC::ClipStationProDeviceC(const StringC &devName,const ImageRectangleC &nrect)
    : dev(0),
      fifo(0),
      rect(nrect),
      useDMA(true),
      doInput(true),
    fifoMode(true)
  {
    dev = sv_open((char *) devName.chars());
    ONDEBUG(cerr << "CSP Device open:" << ((void *) dev) << " (" << devName << ")\n");
    Init();
  }
  
  ClipStationProDeviceC::~ClipStationProDeviceC() {
    ONDEBUG(cerr << "CSP Closing device.\n");
    if(fifo != 0) {
      sv_fifo_free(dev,fifo);
      fifo = 0;
    }
    if(dev != 0) {
      sv_close(dev);
      dev = 0;
    }
    dev = 0;
  }
  
  //: Setup video modes to a suitable default.
  
  bool ClipStationProDeviceC::Init() {
    //sv_info svinfo;
    //  sv_status(dev,&svinfo);
    
    int ret;
    if((ret = sv_videomode(dev,SV_MODE_PAL )) != SV_OK) {
      cerr << "Failed to set mode to PAL. Error:" << ret <<"\n";
      return false;
    }
    if((ret = sv_option(dev,SV_OPTION_LOOPMODE,SV_LOOPMODE_INFINITE)) != SV_OK) {
      cerr << "Failed to set loopmode to inifite:" << ret << "\n";
      return false;
    }
    
    if((ret = sv_sync(dev,SV_SYNC_EXTERNAL)) != SV_OK) {
      cerr << "Failed to set sync to external:" << ret << "\n";
      return false;
    }
    
#if 0
    if((ret = sv_option(dev,SV_OPTION_INPUTPORT,SV_INPORT_SDI)) != SV_OK) {
      cerr << "Failed to set input to SDI 1:" << ret << "\n";
      return false;
    }
    if((ret = sv_preset(dev,SV_PRESET_VIDEO)) != SV_OK) {
      cerr << "Failed to preset video. Error:" << ret << "\n";
      return false;
    }    
#endif 
    ONDEBUG(cerr << "videomode returned:" << ret << "\n");
    if(fifoMode) {
      //ret = sv_fifo_init(dev,&fifo,1,1,0,0,0);
      ret = sv_fifo_init(dev,&fifo,1,1,useDMA,0,0);
      //fifo.
      ONDEBUG(cerr << "Fifo_init:" << ret << "\n" << sv_geterrortext(ret) << "\n");
      ret = sv_fifo_start(dev,fifo); // Start it going...
      ONDEBUG(cerr << "Fifo_start:" << ret << "\n" << sv_geterrortext(ret) << "\n");
    }
    return true;
  }
  
  //: Get one frame of video.
  
  bool ClipStationProDeviceC::GetFrame(void *buff,int x,int y) {
    int ret;
    if(dev == 0)
      return false;
    if(!fifoMode) {
      ret = sv_sv2host(dev,(char *) buff,x*y * 2+ 1000,x,y,0,1,SV_TYPE_YUV422| SV_DATASIZE_8BIT); //
      //ret = sv_record(dev,(char *) buff,x*y * 2,&x,&y,0,1,0);
      ONDEBUG(cerr << "sv2host return:" << ret << "\n");
      int mid = x*y;
      for(int i = mid ;i < (mid + 100) ;i++)
	cerr << (int) ((char *) buff)[i] << ' ';
      return true;
    }
    sv_fifo_buffer *svbuf;
    //sv_fifo_bufferinfo bufinfo;
    if((ret = sv_fifo_getbuffer(dev,fifo,&svbuf,0,SV_FIFO_FLAG_VIDEOONLY)) != SV_OK) {
      ONDEBUG(cerr << "Failed to get frame :" << ret << "\n");
      return false;
    }
    if(svbuf->video[0].size <= (x * y * 2)) {
      //RavlAssert(svbuf->video[0].addr != 0);
      memcpy(buff,svbuf->video[0].addr,svbuf->video[0].size);
    } else
      cerr << "ERROR ClipStationProDeviceC: Buffer to big. \n";
    if((ret = sv_fifo_putbuffer(dev,fifo,svbuf,0)) != SV_OK) {
      cerr << "ERROR ClipStationProDeviceC: Failed to put frame :" << ret << "\n";
      return false;
    }
    ONDEBUG(cerr << "GotFrame.\n");
    return true;
  }
  
  //: Get one field of video.
  
  BufferC<ByteYUV422ValueC> ClipStationProDeviceC::GetFrame() {
    sv_fifo_buffer *svbuf;
    //sv_fifo_bufferinfo bufinfo;
    if(dev == 0) {
      cerr << "ClipStationProDeviceC::GetFrame(), ERROR: Invalid handle. \n";
      return BufferC<ByteYUV422ValueC>();
    }
    int ret;
    if((ret = sv_fifo_getbuffer(dev,fifo,&svbuf,0,SV_FIFO_FLAG_VIDEOONLY)) != SV_OK) {
      cerr << "Failed to get frame :" << ret << "\n";
      return BufferC<ByteYUV422ValueC>();
    }
    
    DMABufferC<ByteYUV422ValueC> buf((svbuf->video[0].size + svbuf->video[1].size) / sizeof(ByteYUV422ValueC) );
    //cerr << "field size: " << svbuf->video[0].size << "\n";
    
    if(useDMA) {
      svbuf->dma.addr = (char *)  buf.ReferenceElm();
      svbuf->dma.size = svbuf->video[0].size + svbuf->video[1].size;
      //cerr << "Doing DMA\n";
    } else {
      //cerr << "Doing memcpy.\n";
      memcpy(buf.ReferenceElm(),svbuf->video[0].addr,svbuf->video[0].size + svbuf->video[1].size);
    }
    if((ret = sv_fifo_putbuffer(dev,fifo,svbuf,0)) != SV_OK) {
      cerr << "ERROR ClipStationProDeviceC: Failed to put frame " << ret << "\n";
      return DMABufferC<ByteYUV422ValueC>();
    }
#if DODEBUG
    sv_fifo_info info;
    sv_fifo_status(dev,fifo,&info);
    cerr << "Got frame. Dopped:" << info.dropped << " \n";
#endif
    return buf;
  }
  
  
  //: Get one frame of video.
  
  bool ClipStationProDeviceC::PutFrame(void *buff,int x,int y) {
    int ret;
    if(!fifoMode) {
      ret = sv_host2sv(dev,(char *) buff,x*y * 2,x,y,0,1,SV_TYPE_YUV422);
      return true;
    }
    return true;
  }

}
