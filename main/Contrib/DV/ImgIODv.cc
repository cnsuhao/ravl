// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2002, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlDV
//! file="Ravl/Contrib/DV/ImgIODv.cc"

#include "Ravl/Image/ImgIODv.hh"
#include "Ravl/OS/Filename.hh"

namespace RavlImageN {

  //: Constructor from stream 
  DPIImageDvBodyC::DPIImageDvBodyC(const IStreamC &nStrm,const StringC &suffix)
    : DPImageDvBaseBodyC(suffix),
      strm(nStrm), 
      dv(true)
  {
 
    //  tcGrab = TimeCodeC((long int)0);

    if(!strm)
      cerr << "DPIImageDvBodyC::DPIImageDvBodyC(IStreamC), Passed bad stream. \n";
    else {
      if(!nStrm.Name().IsEmpty()) {
	FilenameC tmp(nStrm.Name());
	// Stream must be good, so if file doesn't exist maybe its
	// some kind of named pipe.
	if(tmp.Exists())
	  SetSequenceSize (tmp.FileSize() / frameSize);
      }
    }
  }


  ///////////////////////////
  //: Seek to location in stream.
  // Returns false, if seek failed. (Maybe because its
  // not implemented.)
  // if an error occurered (Seek returned False) then stream
  // position will not be changed.
  
  bool DPIImageDvBodyC::Seek(UIntT off) {
    if(off == ((UIntT) -1))
      return false; // File to big.
    strm.is().clear(); // Clear any errors.
    strm.Seek(CalcOffset(off));
    frameNo = off;// Wait to after seek in case of exception.
    return true;
  }

  //: Delta Seek, goto location relative to the current one.

  bool DPIImageDvBodyC::DSeek(IntT off) {
    if(off < 0) {
      if((-off) > (IntT) frameNo)
	return false; // Seek off begining of data.
    }
    UIntT nfrmno = frameNo + off;
    strm.Seek(CalcOffset(nfrmno));
    frameNo = nfrmno; // Wait to after seek in case of exception.
    return true;
  }

  //: Find current location in stream.
  UIntT DPIImageDvBodyC::Tell() const
  { return frameNo; }

  //: Find the total size of the stream.
  UIntT DPIImageDvBodyC::Size() const
  { return SeqSize(); }
  
  /////////////////////////
  //: Get next image.
  
  ImageC<ByteRGBValueC> DPIImageDvBodyC::Get() {
    ImageC<ByteRGBValueC> head;
    Get(head);
    return head;
  }

  //////////////////////////
  //: Get next image.

  bool DPIImageDvBodyC::Get(ImageC<ByteRGBValueC> &head)
  { 
    //: check the stream
    if(!strm.good())
      return false;
  
    //: read the data
    char data[144000];
    strm.read(data, 144000);

    //: decode it
    head = dv.Apply((ByteT*)&data[0]);
    
    //frame.setData((ByteT*)&data[0]);
    //head = frame.Image();
    //tcGrab = frame.extractTimeCode();
    //cout << head[10][10] << " : " << tcGrab << endl;
    frameNo++;
    return true; 
  }
  
  //: Get a stream attribute.
  // Returns false if the attribute name is unknown.
  // This is for handling stream attributes such as frame rate, and compression ratios.
  bool DPIImageDvBodyC::GetAttr(const StringC &attrName,StringC &attrValue) {
    if(attrName == "deinterlace") {
      attrValue = dv.Deinterlace() ? "1" : "0";
      return true;
    }
    return DPISPortBodyC<ImageC<ByteRGBValueC> >::GetAttr(attrName,attrValue);
  }
  
  //: Set a stream attribute.
  // Returns false if the attribute name is unknown.
  // This is for handling stream attributes such as frame rate, and compression ratios.
  
  bool DPIImageDvBodyC::SetAttr(const StringC &attrName,const StringC &attrValue) {
    if(attrName == "deinterlace") {
      if(attrValue == "1")
	dv.Deinterlace(true);
      else
	dv.Deinterlace(false);
      return true;
    }
    return DPISPortBodyC<ImageC<ByteRGBValueC> >::SetAttr(attrName,attrValue);
  }


  //: Constructor from filename.  

  DPIImageDvC::DPIImageDvC(const StringC &fn)
    : DPEntityC(*new DPIImageDvBodyC(IStreamC(fn),FilenameC(fn).Extension()))
  {}

  //: Constructor from filename.  

} // end namespace
