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
#include "Ravl/DVDRead.hh"
#include "Ravl/IO.hh"
#include "Ravl/OS/Date.hh"
#include "Ravl/DList.hh"

using namespace std;
using namespace RavlN;

int main(int nargs,char **argv)
{
  OptionC opts(nargs,argv);
  StringC device = opts.String("d", "/dev/dvd", "DVD device.");
  IntT title = opts.Int("t", 1, "DVD title.");
  opts.Check();
  
  // Create the dvd 
  DVDReadC dvd(title, device);

  // Display the stream attributes
  DListC<StringC> attrs;
  if (dvd.GetAttrList(attrs))
  {
    DLIterC<StringC> it(attrs);
    while (it)
    {
      StringC value;
      if (dvd.GetAttr(*it, value))
        cerr << *it << " : " << value << endl;
      it++;
    }
  }
  
  // Create the output file
  StringC filename = StringC(title) + ".vob";
  OStreamC file(filename);

  // Load the stream
  SArray1dC<ByteT> data(1024 * 1024);
  UIntT size = 0;
  while((size = dvd.GetArray(data)) > 0)
  {
    cerr << "tell(" << dvd.Tell64() << ")" << endl;
    file.write(reinterpret_cast<const char*>(&(data[0])), size);
  }

  file.Close();
  
  return 0;
}
