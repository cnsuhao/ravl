// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlImgIO1394dc


#include "Ravl/Option.hh"
#include "Ravl/Image/ImgIO1394dc.hh"
#include "Ravl/IO.hh"
#include "Ravl/DP/SequenceIO.hh"

using namespace RavlN;
using namespace RavlImageN;

int main(int nargs,char **argv) {
  OptionC opt(nargs,argv);
  StringC dev = opt.String("d","/dev/raw1394","Firewire device. ");
  StringC out = opt.String("o","@X","Output sequence");
  IntT n = opt.Int("n",-1,"Number of frames (default: unlimited)");
  opt.Check();
  
  ImgIO1394dcBaseC imgio;
  if(!imgio.Open(dev,typeid(ByteT))) {
    cerr << "Failed to setup camera. \n";
    return 1;
  }
  DPOPortC<ImageC<ByteT> > imgOut;
  if(!OpenOSequence(imgOut,out)) {
    cerr << "Failed to open output. \n";
    return 0;
  }
  for(IntT i(0); i!=n; ++i) {
    ImageC<ByteT> img;
    imgio.CaptureImage(img);
    imgOut.Put(img);
  }
  
  return 0;
}
