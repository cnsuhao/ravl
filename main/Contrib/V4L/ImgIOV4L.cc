// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
///////////////////////////////////////////////////////////
//! rcsid="$Id$"
//! lib=RavlImgIOV4L
//! file="Contrib/V4L/ImgIOV4L.cc"

typedef unsigned long ulong;

#define USE_PHILIPS_WEBCAM 0

#include <linux/videodev.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <string.h>

#if USE_PHILIPS_WEBCAM
#include "pwc-ioctl.h"
#endif

#include "Ravl/Image/Image.hh"
#include "Ravl/DP/FileFormatIO.hh"
#include "Ravl/Image/SubSample.hh"
#include "Ravl/Image/ImgIOV4L.hh"
#include "Ravl/Array2dIter.hh"

#define DODEBUG 0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

#define USE_MMAP 1

namespace RavlImageN {
  
  //: Constructor.
  
  DPIImageBaseV4LBodyC::DPIImageBaseV4LBodyC(const StringC &dev,const type_info &npixType,const ImageRectangleC &nrect)
    : rect(nrect),
      fd(-1),
      buf_grey(0),
      buf_u(0),
      buf_v(0),
      half(false)
  {
    sourceType = SOURCE_UNKNOWN;
    if(!Open(dev,npixType,nrect))
      cerr << "DPIImageBaseV4LBodyC::DPIImageBaseV4LBodyC(), Failed to  open '" << dev << "'\n";
    //DumpParam(cerr);
  }

  DPIImageBaseV4LBodyC::DPIImageBaseV4LBodyC(const StringC &dev,const type_info &npixType,bool nhalf)
    : rect(0,-1,0,-1),
      fd(-1),
      buf_grey(0),
      buf_u(0),
      buf_v(0),
      half(nhalf)
  {
    sourceType = SOURCE_UNKNOWN;
    if(!Open(dev,npixType,rect))
      cerr << "DPIImageBaseV4LBodyC::DPIImageBaseV4LBodyC(), Failed to  open '" << dev << "'\n";
    //DumpParam(cerr);
  }
  
  //: Destructor.
  
  DPIImageBaseV4LBodyC::~DPIImageBaseV4LBodyC() {
    Close();
  }
  
  //: Check what video channels are available.
  
  bool DPIImageBaseV4LBodyC::CheckChannels(int nchannels) {
    video_channel vidchan;
    for(int i = 0;i < nchannels;i++) {
      vidchan.channel = i;
      if(ioctl(fd,VIDIOCGCHAN,&vidchan) < 0) {
	cerr << "Failed to get info on video channel. \n";
	return false;
      }
      cerr << "Channel=" << vidchan.channel << " name=" << vidchan.name << " \n";
      cerr << "tunners=" << vidchan.tuners << " flags=" << vidchan.flags << " \n";
      cerr << "type=" << vidchan.type << " norm=" << vidchan.norm << " \n";
    }
    
    vidchan.channel = 1;
    if(ioctl(fd,VIDIOCGCHAN,&vidchan) < 0) {
      cerr << "Failed to get info on video channel. \n";
      return false;
    }
    vidchan.channel = 1;
    vidchan.norm = 1;
    if(ioctl(fd,VIDIOCSCHAN,&vidchan) < 0) {
      cerr << "Failed to set video channel. \n";
      return false;
    }
    return true;
  }

  //: Open a meteor device.
  
  bool DPIImageBaseV4LBodyC::Open(const StringC &dev,const type_info &npixType,const ImageRectangleC &nrect) {
    
    fd = open(dev.chars(),O_RDWR);
    if(fd == 0) {
      cerr << "Failed to open video device. '" << dev << "' \n";
      return false;
    }
    
    struct video_capability vidcap;
    if(ioctl(fd,VIDIOCGCAP,&vidcap) < 0) {
      cerr << "Failed to get video capabilities. \n";
      return 1;
    }
    ONDEBUG(cerr << "Video type '" << vidcap.name << "' Type=" << vidcap.type << "\n");
    ONDEBUG(cerr << "Channels=" << vidcap.channels << " Audios=" << vidcap.audios << "\n");
    ONDEBUG(cerr << "MaxWidth=" << vidcap.maxwidth << " MaxHeight=" << vidcap.maxheight << "\n");
    ONDEBUG(cerr << "MinWidth=" << vidcap.minwidth << " MinHeight=" << vidcap.minheight << "\n");

    if(vidcap.channels > 1)
      CheckChannels(vidcap.channels);
    
    sourceType = SOURCE_UNKNOWN;
    //int camtype;
    if(strncmp(vidcap.name,"Philips ",8) == 0) {
      ONDEBUG(cerr << "Got philips webcam. \n");
      sourceType = SOURCE_USBWEBCAM_PHILIPS;
    }
    // Setup capture mode.
    
    struct video_picture vidpic;
    if(ioctl(fd,VIDIOCGPICT,&vidpic) < 0) {
      cerr << "Failed to get video picture paramiters. \n";
      return 1;
    }

    if(npixType == typeid(ByteYUVValueC)) {
      vidpic.palette = VIDEO_PALETTE_UYVY;
      if(ioctl(fd,VIDIOCSPICT,&vidpic) < 0) {
	vidpic.palette = VIDEO_PALETTE_YUYV;
	if(ioctl(fd,VIDIOCSPICT,&vidpic) < 0) {
	  vidpic.palette = VIDEO_PALETTE_YUV420P;
	  if(ioctl(fd,VIDIOCSPICT,&vidpic) < 0) {
	    cerr << "Failed to set video picture paramiters. \n";
	    return 1;
	  }
	}
      }
    } else if(npixType == typeid(ByteRGBValueC)) {
      vidpic.palette = VIDEO_PALETTE_RGB24;
      if(ioctl(fd,VIDIOCSPICT,&vidpic) < 0) {
	vidpic.palette = VIDEO_PALETTE_RGB24 | 0x80;
	if(ioctl(fd,VIDIOCSPICT,&vidpic) < 0) {
	  cerr << "Failed to set video picture paramiters. \n";
	  return 1;	
	}
      }
    }
    if(ioctl(fd,VIDIOCGPICT,&vidpic)) {
      cerr << "Failed to get video mode. \n";
      return 1;
    }
    palette = vidpic.palette;
#if DODEBUG
    switch(palette) {
    case VIDEO_PALETTE_RGB24: 
      cerr << "Using pallete VIDEO_PALETTE_RGB24 \n";
      break;
    case VIDEO_PALETTE_RGB24 | 0x80:
      cerr << "Using pallete VIDEO_PALETTE_BGR24 \n";
      break;
    case VIDEO_PALETTE_YUYV:
      cerr << "Using pallete VIDEO_PALETTE_YUYV \n";
      break;
    case VIDEO_PALETTE_UYVY:
      cerr << "Using pallete VIDEO_PALETTE_UYVY \n";
      break;
    case VIDEO_PALETTE_YUV420P:
      cerr << "Using pallete VIDEO_PALETTE_YUV420P \n";
      break;
    default:
      cerr << "Using pallete " << palette << "\n";
    }
#endif
    
    struct video_window vidwin;
    vidwin.x = 0;
    vidwin.y = 0;
    if(!half) {
      vidwin.width = vidcap.maxwidth;
      vidwin.height = vidcap.maxheight;
    } else {
      vidwin.width = vidcap.maxwidth /2;
      vidwin.height = vidcap.maxheight/2;
    }
    vidwin.chromakey = 0;
#if USE_PHILIPS_WEBCAM
    if(sourceType == SOURCE_USBWEBCAM_PHILIPS)
      vidwin.flags = (5 << PWC_FPS_SHIFT);// | PWC_FPS_SNAPSHOT;
    else
#endif
      vidwin.flags = 0;
    vidwin.clips = 0;
    vidwin.clipcount = 0;
    
    if(ioctl(fd,VIDIOCSWIN,&vidwin) < 0) {    
      cerr << "WARNING: Failed to set video window paramiters. \n";
    }
    if(ioctl(fd,VIDIOCGWIN,&vidwin) < 0) {    
      cerr << "WARNING: Failed to get video window paramiters. \n";
    }
    ONDEBUG(cerr << "Capture:\n");
    ONDEBUG(cerr << "X=" << vidwin.x << " Y=" << vidwin.y << "\n");
    ONDEBUG(cerr << "Width=" << vidwin.width << " MinHeight=" << vidwin.height << "\n");
    ONDEBUG(cerr << "Rate=" << ((vidwin.flags & PWC_FPS_FRMASK) >> PWC_FPS_SHIFT)  << "\n");
    rect = ImageRectangleC(0,vidwin.height-1,
			   0,vidwin.width -1
			   );
    
    switch(sourceType) {
    case SOURCE_USBWEBCAM_PHILIPS: SetupPhilips(); break;
    case SOURCE_UNKNOWN: break;
    }
    
    switch(palette)
      {
      case VIDEO_PALETTE_YUV420P:
	buf_grey = new ByteT [rect.Area()];
	buf_u = new ByteT [rect.Area()/4];
	buf_v = new ByteT [rect.Area()/4];
	break;
      case VIDEO_PALETTE_UYVY:
      case VIDEO_PALETTE_YUYV:
	buf_grey = new ByteT [rect.Area() * 2];
	buf_u = 0;
	buf_v = 0;
	break;
      default:
	buf_grey = 0;
	buf_u = 0;
	buf_v = 0;
	break;
      }
    
    return true;
  }
  
  //: Setup a philips webcam.
  
  bool DPIImageBaseV4LBodyC::SetupPhilips() {
#if USE_PHILIPS_WEBCAM
    ONDEBUG(cerr << "DPIImageBaseV4LBodyC::SetupPhilips(), Called \n");
    // Setup colour balance.
    pwc_whitebalance wb;
    if(ioctl(fd,VIDIOCPWCGAWB,&wb) < 0) {
      cerr<< "WARNING: Failed to get while balance. \n";
    }
    wb.mode = PWC_WB_FL; //PWC_WB_OUTDOOR; //
    if(ioctl(fd,VIDIOCPWCSAWB,&wb) < 0) {
      cerr<< "WARNING: Failed to set while balance. \n";
    }
#endif
    return true;
  }
  
  
  
  bool DPIImageBaseV4LBodyC::Close() {
    if(fd < 0)
      return true; // Already closed.
    close(fd);
    fd = -1;
    delete buf_grey;
    delete buf_u;
    delete buf_v;
    return true;
  }
  
  bool DPIImageBaseV4LBodyC::NextFrame(ImageC<ByteYUVValueC> &ret) {
    ONDEBUG(cerr << "DPIImageBaseV4LBodyC::NextFrame(ImageC<ByteYUVValueC>&) Called \n");
    if(fd < 0) {
      cerr << "ERROR: No filehandle \n";
      ret =  ImageC<ByteYUVValueC>();
      return false;
    }
    int rsize,rret;
    
    switch(palette) 
      {
      case VIDEO_PALETTE_YUV420P: 
	{
	  rsize = rect.Area();
	  if((rret = read(fd,buf_grey,rsize)) != rsize) {
	    cerr << "Read failed. Bytes read = " << rret << " \n";
	    return false;
	  }
	  rsize /= 4;
	  if((rret = read(fd,buf_u,rsize)) != rsize) {
	    cerr << "Read failed. Bytes read = " << rret << "\n";
	    return false;
	  }
	  if((rret = read(fd,buf_v,rsize)) != rsize) {
	    cerr << "Read failed. Bytes read = " << rret << "\n";
	    return false;
	  }
	  ret = ImageC<ByteYUVValueC>(rect);
	  ByteT *gat = buf_grey;
	  ByteT *uat = buf_u;
	  ByteT *vat = buf_v;
	  ByteT *urow = buf_u;
	  ByteT *vrow = buf_v;
	  for(Array2dIterC<ByteYUVValueC> it(ret);it;) {
	    uat = urow;
	    vat = vrow;
	    do {
	      *it = ByteYUVValueC(*gat,*uat+128,*vat+128);
	      it++; gat++;
	      *it = ByteYUVValueC(*gat,*uat+128,*vat+128);
	      gat++; uat++; vat++;
	    } while(it.Next());
	    uat = urow;
	    vat = vrow;
	    do {
	      *it = ByteYUVValueC(*gat,*uat+128,*vat+128);
	      it++; gat++;
	      *it = ByteYUVValueC(*gat,*uat+128,*vat+128);
	      gat++; uat++; vat++;
	    } while(it.Next());
	    urow += rect.Cols()/2;
	    vrow += rect.Cols()/2;
	  }
	}
	break;
      case VIDEO_PALETTE_YUYV: 
	{
	  RavlAssert(buf_grey != 0);
	  ret = ImageC<ByteYUVValueC>(rect);
	  rsize = rect.Area() * 2;
	  if((rret = read(fd,buf_grey,rsize)) != rsize) {
	    cerr << "Read failed. Bytes read = " << rret << "\n";
	    return false;
	  }
	  ByteT *gat = buf_grey;
	  for(Array2dIterC<ByteYUVValueC> it(ret);it;) {
	    ByteT y1 = *gat++;
	    ByteT u = (*gat++) + 128;
	    ByteT y2 = *gat++;
	    ByteT v = (*gat++) + 128;
	    *it = ByteYUVValueC(y1,u,v);
	    it++;
	    *it = ByteYUVValueC(y2,u,v);
	    it++;
	  }
	}
	break;
      case VIDEO_PALETTE_UYVY:
	{
	  RavlAssert(buf_grey != 0);
	  ret = ImageC<ByteYUVValueC>(rect);
	  rsize = rect.Area() * 2;
	  if((rret = read(fd,buf_grey,rsize)) != rsize) {
	    cerr << "Read failed. Bytes read = " << rret << "\n";
	    return false;
	  }
	  ByteT *gat = buf_grey;
	  for(Array2dIterC<ByteYUVValueC> it(ret);it;) {
	    ByteT u = (*gat++) + 128;
	    ByteT y1 = *gat++;
	    ByteT v = (*gat++) + 128;
	    ByteT y2 = *gat++;
	    *it = ByteYUVValueC(y1,u,v);
	    it++;
	    *it = ByteYUVValueC(y2,u,v);
	    it++;
	  }
	}
	break;
      default:
	cerr << "DPIImageBaseV4LBodyC::NextFrame(ImageC<ByteYUVValueC>), Don't know how to handle palette mode: " << palette << "\n";
	return false;
      }
    return true;
  }
  

  //: Get next RGB frame from grabber.
  
  bool DPIImageBaseV4LBodyC::NextFrame(ImageC<ByteRGBValueC> &ret) {
    ONDEBUG(cerr << "DPIImageBaseV4LBodyC::NextFrame(ImageC<ByteRGBValueC>&) Called \n");
    if(fd < 0) {
      cerr << "ERROR: No filehandle \n";
      ret =  ImageC<ByteRGBValueC>();
      return false;
    }
    switch(palette) 
      {
      case VIDEO_PALETTE_RGB24:
	ret = ImageC<ByteRGBValueC>(rect);
	read(fd,&(ret[rect.Origin()]),rect.Area());
	break;
      case VIDEO_PALETTE_RGB24 | 0x80:
	ret = ImageC<ByteRGBValueC>(rect);
	read(fd,&(ret[rect.Origin()]),rect.Area());
	// Swap blue and red.
	for(Array2dIterC<ByteRGBValueC> it(ret);it;it++) {
	  ByteT x = it.Data().Blue();
	  it.Data().Blue() = it.Data().Red();
	  it.Data().Red() = x;
	}
	break;
      default:
	cerr << "DPIImageBaseV4LBodyC::NextFrame(ImageC<ByteRGBValueC>), Don't know how to handle palette mode: " << palette << "\n";
	return false;
      }
    return true;
  }
  
}

