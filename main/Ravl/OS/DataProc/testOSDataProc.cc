// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2002, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlDPMT


#include "Ravl/DP/ThreadPipe.hh"
#include "Ravl/DP/MTIOConnect.hh"

using namespace RavlN;

template class DPThreadPipeC<IntT,IntT>;
template class DPMTIOConnectC<IntT>;

int testThreadPipe();

int main() {
  int ln;
  if((ln = testThreadPipe()) != 0) {
    cerr << "Error on line " << ln << "\n";
    return 1;
  }
  cerr << "Test passed. \n";
  return 0;
}

int testThreadPipe() {
  //...
  return 0;
}
