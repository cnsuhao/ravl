// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlDPGraph
//! file="Ravl/GUI/DPDisplay/exDPWindow.cc"
//! author="Charles Galambos"
//! docentry="Ravl.GUI.Data Display"
//! userlevel=Normal

#include "Ravl/Array1dIter.hh"
#include "Ravl/IO.hh"
#include "Ravl/StdConst.hh"

using namespace RavlN;

int main() {
  
  Array1dC<RealT> data(100);
  RealT val = 0;
  for(Array1dIterC<RealT> it(data);it;it++,val += (RavlConstN::pi/40))
    *it = Sin(val);
  
  if(!Save("@GRAPH:exDPGraphWindow",data,"",true)) {
    cerr << "Failed to save image. \n";
    return 1;
  }
  
  return 0;
}
