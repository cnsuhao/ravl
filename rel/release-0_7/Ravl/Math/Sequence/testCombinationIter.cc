// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
/////////////////////////////
//! rcsid="$Id$"
//! lib=RavlMath
//! file="Ravl/Math/Sequence/testCombinationIter.cc"

#include "Ravl/CombinationIter.hh"
#include "Ravl/Stream.hh"

using namespace RavlN;

int main(int nargs,char **argv) {
  DListC<IntT> data;
  data.InsLast(1);
  data.InsLast(2);
  data.InsLast(3);
  data.InsLast(4);
  cout << "Starting CombinationIterC test...\n";
  IntT n = 0;
  for(CombinationIterC<IntT> it(data);it.IsElm();it.Next()) {
    for(DLIterC<IntT> dit(it.Data());dit.IsElm();dit.Next())
      cout << " " << dit.Data();
    cout << endl;
    n++;
  }
  if(n != 15) {
    cerr << "CombinationIterC test failed. \n";
    return 1;
  }
  n = 0;
  cout << "Test 1 passed. \n";
  for(CombinationIterC<IntT> it(data,2);it.IsElm();it.Next()) {
    for(DLIterC<IntT> dit(it.Data());dit.IsElm();dit.Next())
      cout << " " << dit.Data();
    cout << endl;
    n++;
  }
  if(n != 11) {
    cerr << "CombinationIterC test failed. \n";
    return 1;
  }
  cout << "Test 2 passed. \n";
  return 0;
}
