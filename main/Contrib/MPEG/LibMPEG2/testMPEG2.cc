// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//////////////////////////////////////////////////////////////////
//! rcsid = "$Id$"
//! lib = RavlLibMPEG2
//! author = "Warren Moore"

#include "Ravl/Option.hh"
#include "Ravl/Image/LibMPEG2Format.hh"
#include "Ravl/IO.hh"
#include "Ravl/DP/SPort.hh"
#include "Ravl/OS/Date.hh"
#include "Ravl/OS/Filename.hh"

using namespace RavlN;
using namespace RavlImageN;

int main(int nargs, char **argv)
{
  OptionC opts(nargs,argv);
  StringC filename = opts.String("", "in.mpeg", "Input mpeg file.");
  opts.Check();
  
  // Check the file exists
  FilenameC fn(filename);
  if (!fn.Exists())
  {
    cerr << "Error opening file (" << filename << ")\n";
    return 1;
  }

  // Select the correct opening method
  FileFormatLibMPEG2C format;
  DPISPortC< ImageC<ByteRGBValueC> > in = format.CreateInput(filename);
  if (!in.IsValid())
  {
    cerr << "Unable to open file (" << filename << ")\n";
    return 1;
  }

  // Load the stream
  ImageC<ByteRGBValueC> rgb;
  UIntT i = 0;
  while(1)
  {
    if(!in.Get(rgb))
      break;
    
    i++;
#if 0
    if(i > 30)
    {
      in.Seek(0);
      i = 0;
    }
#endif
    RavlN::Save("@X", rgb);
    Sleep(0.1);
  }

  return 0;
}
