// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//////////////////////////////////////////////////////////////////
//! rcsid = "$Id$"
//! lib = RavlDVDRead
//! author = "Warren Moore"

#include "Ravl/Option.hh"
#include "Ravl/Image/ImgIOMPEG2.hh"
#include "Ravl/DVDRead.hh"
#include "Ravl/IO.hh"
#include "Ravl/DP/SPort.hh"
#include "Ravl/DP/SPortAttach.hh"
#include "Ravl/OS/Date.hh"
#include <fstream>

using namespace RavlN;
using namespace RavlImageN;

int main(int nargs,char **argv)
{
  OptionC opts(nargs,argv);
  StringC device = opts.String("d", "/dev/dvd", "DVD device.");
  IntT title = opts.Int("t", 1, "DVD title.");
  opts.Check();
  
  // Create the dvd 
  DVDReadC dvd(title, device);

  // Create the DP path
  DPISPortC< ImageC<ByteRGBValueC> > in(SPort(dvd >> ImgILibMPEG2C(0xe0)));

  // Load the stream
  ImageC<ByteRGBValueC> rgb;
  UIntT i = 0;
  while(1)
  {
    if(!in.Get(rgb))
      break;
    
    i += 25;
    in.Seek(i);
#if 0
    if(i > 30)
    {
      in.Seek(0);
      i = 0;
    }
#endif
    RavlN::Save("@X",rgb);
    Sleep(0.1);
  }

  return 0;
}
