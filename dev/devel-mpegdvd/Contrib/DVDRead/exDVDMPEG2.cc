// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU
// General Public License (GPL). See the gpl.licence file for details or
// see http://www.gnu.org/copyleft/gpl.html
// file-header-ends-here
//////////////////////////////////////////////////////////////////
//! rcsid = "$Id$"
//! lib = RavlDVDRead
//! author = "Warren Moore"

#include "Ravl/Option.hh"
#include "Ravl/DVDFormat.hh"
#include "Ravl/IO.hh"
#include "Ravl/DP/SPort.hh"
#include "Ravl/OS/Date.hh"
#include "Ravl/OS/Filename.hh"
#include "Ravl/Random.hh"

using namespace RavlN;
using namespace RavlImageN;

int main(int nargs, char **argv)
{
  OptionC opts(nargs,argv);
  StringC device = opts.String("d", "/dev/dvd", "DVD device.");
  IntT title = opts.Int("t", 1, "DVD title.");
  opts.Check();
  
  // Create the option string
  StringC dvdString("@DVD:");
  dvdString += StringC(title);
  dvdString += "#" + device;
  
  // Select the correct opening method
  FileFormatDVDC format;
  DPIPortC< ImageC<ByteRGBValueC> > in = format.CreateInput(dvdString);
  if (!in.IsValid())
  {
    cerr << "Unable to open DVD device (" << dvdString << ")\n";
    return 1;
  }

  // Display the stream attributes
  DListC<StringC> attrs;
  if (in.GetAttrList(attrs))
  {
    DLIterC<StringC> it(attrs);
    while (it)
    {
      StringC value;
      if (in.GetAttr(*it, value))
        cerr << *it << " : " << value << endl;
      it++;
    }
  }
  
  // Delay in seconds
  const RealT delay = 0.010;
  
  // Load the stream
  ImageC<ByteRGBValueC> rgb;
  
  // Keep playing til the end
  while (!in.IsGetEOS())
  {
    if(!in.Get(rgb))
      break;
    
    RavlN::Save("@X", rgb);

    Sleep(delay);
  }

  return 0;
}
