// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid = "$Id$"
//! lib = RavlDVDRead
//! author = "Warren Moore"

#include "Ravl/Option.hh"
#include "Ravl/DVDRead.hh"
#include "Ravl/IO.hh"
#include "Ravl/OS/Date.hh"
#include <fstream>

using namespace std;
using namespace RavlN;

int main(int nargs,char **argv) {
  OptionC opts(nargs,argv);
  StringC device = opts.String("d", "/dev/dvd", "DVD device.");
  IntT title = opts.Int("t", 1, "DVD title.");
  opts.Check();
  
  // Create the dvd 
  DVDReadC dvd(title, device);

  // Create the output file
  StringC filename = StringC(title) + ".vob";
  ofstream file(filename, ios::out|ios::binary);

  // Load the stream
  SArray1dC<ByteT> data(1024 * 1024);
  UIntT size = 0;
  while((size = dvd.GetArray(data)) > 0)
  {
    file.write(reinterpret_cast<const char*>(&(data[0])), size);
  }

  file.close();
  
  return 0;
}
