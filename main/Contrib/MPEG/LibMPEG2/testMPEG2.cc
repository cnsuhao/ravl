// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlLibMPEG2

#include "Ravl/Option.hh"
#include "Ravl/Image/ImgIOMPEG2.hh"
#include "Ravl/IO.hh"

using namespace RavlN;
using namespace RavlImageN;


int main(int nargs,char **argv) {
  OptionC opts(nargs,argv);
  StringC inFile = opts.String("","in.mpeg","Input mpeg file. ");
  opts.Check();
  
  IStreamC strm(inFile);
  if(!strm) {
    cerr << "Error opening file " << inFile << "\n";
    return 1;
  }
  ImgILibMPEG2C mi(strm);
  ImageC<ByteRGBValueC> rgb;
  UIntT i = 0;
  while(1) {
    //cerr << "Reading image... \n";
    if(!mi.Get(rgb))
      break;
    i++;
#if 0
    if(i > 40) {
      mi.Seek(0);
      i = 0;
    }
#endif
    RavlN::Save("@X",rgb);

  }
  return 0;
}
