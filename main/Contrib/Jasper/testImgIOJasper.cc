// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2005, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! author="Charles Galambos"

#include "Ravl/Image/ImgIOJasper.hh"
#include "Ravl/BufStream.hh"
#include "Ravl/Image/DrawFrame.hh"
#include "Ravl/Array2dIter2.hh"

using namespace RavlImageN;

int testStream();

int main(int nargs,char **argv) {
  int ln;
  if((ln = testStream()) != 0) {
    cerr << "Test failed on line " << ln << "\n";
    return 1;
  }
  return 0;
}

int testStream() {

  ImageC<ByteRGBValueC> testImg(100,100);
  ByteRGBValueC back(0,0,0);
  ByteRGBValueC white(255,255,255);
  DrawFrame(testImg,white,IndexRange2dC(20,80,20,80),true);
  cerr << "Coding image. \n";
  SArray1dC<char> data;
  {
    BufOStreamC bis;
    DPOImageJasperByteRGBC imgRGB(bis,"jp2");
    imgRGB.Put(testImg);
    data = bis.Data();
  }
  cerr << "Decoding image. \n";
  ImageC<ByteRGBValueC> outImg;
  {
    BufIStreamC bos(data);
    DPIImageJasperByteRGBC imgRGB(bos);
    if(!imgRGB.Get(outImg)) return __LINE__;
  }
  cerr << "Testing. \n";
  if(outImg.Frame() != testImg.Frame()) return __LINE__;
  for(Array2dIter2C<ByteRGBValueC,ByteRGBValueC> it(testImg,outImg);it;it++) {
    if(Abs(it.Data1().Y() - it.Data2().Y()) > 2) return __LINE__;
  }
  return 0;
}
