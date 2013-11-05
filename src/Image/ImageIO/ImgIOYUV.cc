// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
////////////////////////////////////////////////////////////////
//! rcsid="$Id$"
//! lib=RavlImageIO
//! file="Ravl/Image/ImageIO/ImgIOYUV.cc"

#include "Ravl/BinStream.hh"
#include "Ravl/Image/ImgIOYUV.hh"
#include "Ravl/Array2dIter.hh"
#include "Ravl/OS/Filename.hh"
#include <ctype.h>
#include <stdlib.h>

#define DODEBUG 0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlImageN {
      
  // Get the yuv image size from the file size if available.
  static Index2dC YUVSize(const FilenameC& File) {
    Index2dC vSize;
    if (File.Exists()) {
      switch (File.FileSize()) {
      case 829440:   vSize = Index2dC( 576, 720); break;
      case 4147200:  vSize = Index2dC(1080,1920); break;
      default: throw ExceptionInvalidStreamC(StringC("Unknown yuv file size: ") + (UIntT) File.FileSize() + " bytes");
      }
    }
    return vSize;
  }

  ////////////////////////

  //: Constructor from filename.
  
  DPIImageYUVBodyC::DPIImageYUVBodyC(const StringC & fn)
    : inf(fn,true),
      done(false),
      size(YUVSize(fn))
  {}

  //: Constructor from stream.
  
  DPIImageYUVBodyC::DPIImageYUVBodyC(const IStreamC &in)
    : inf(in),
      done(false),
      size(YUVSize(in.Name()))
  {}

  //: Is valid data ?
  
  bool DPIImageYUVBodyC::IsGetEOS() const
  { return (done || (!inf)) ; }
  
  //: Is some data ready ?
  // true = yes.
  
  bool DPIImageYUVBodyC::IsGetReady() const 
  { return (done || (!inf)); }
  

  //: Get image from stream.
  
  bool DPIImageYUVBodyC::Get(ImageC<ByteYUVValueC> &buff) {
    if(done || (!inf))
      return false;
    try {
      buff = DPIImageYUVBodyC::Get();
    } catch(DataNotReadyC &) {
      return false;
    }
    return true;
  }
  
  //: Get next piece of data.
  
  ImageC<ByteYUVValueC> DPIImageYUVBodyC::Get() {
    if(done)
      throw DataNotReadyC("DPIImageYUVBodyC: Image already read. ");

    ImageC<ByteYUVValueC> img(size[0],size[1]);
    char buff[4];
    for(Array2dIterC<ByteYUVValueC> it(img);it.IsElm();it.Next()) {
      inf.read(buff,4);
      const char u = buff[0] - 128;
      const char v = buff[2] - 128;
      it.Data().U() = u;
      it.Data().V() = v;
      it.Data().Y() = buff[1];
      it.Next();
      it.Data().U() = u;
      it.Data().V() = v;
      it.Data().Y() = buff[3];
    }
    done = true;
    return img;
  }
  

  ///////////////////////////////////////


    //: Constructor from filename.
  
  DPOImageYUVBodyC::DPOImageYUVBodyC(const StringC & fn)
    : outf(fn,true),
      done(false)
  {}
  
  
  //: Constructor from stream.
  
  DPOImageYUVBodyC::DPOImageYUVBodyC(const OStreamC &strm)
    : outf(strm),
      done(false)
  {}
  
  
  //: Is port ready for data ?
  
  bool DPOImageYUVBodyC::IsPutReady() const 
  { return outf && !done;  }
  
  //: Write out image.
  
  bool DPOImageYUVBodyC::Put(const ImageC<ByteYUVValueC> &img) 
  {
    for(Array2dIterC<ByteYUVValueC> it(img);it.IsElm();it.Next()) {
      if (it.Index()[1]%2 != 0)  
        outf.put(it->V()+128);
      else  
        outf.put(it->U()+128);
      outf.put(it->Y());
    }

    outf.os().flush();

    done = true;
    return true;
  }
  

  DPIImageYUVC::DPIImageYUVC(const StringC & fn)
    : DPEntityC(*new DPIImageYUVBodyC(fn))
  {}
  
  
  DPIImageYUVC::DPIImageYUVC(const IStreamC &strm)
    : DPEntityC(*new DPIImageYUVBodyC(strm))
  {}
  
  DPOImageYUVC::DPOImageYUVC(const StringC & fn)
    : DPEntityC(*new DPOImageYUVBodyC(fn))
  {}
  
  
  DPOImageYUVC::DPOImageYUVC(const OStreamC &strm)
    : DPEntityC(*new DPOImageYUVBodyC(strm))
  {}
  

}
