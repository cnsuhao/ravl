// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//////////////////////////////////////////////////////////////
//! rcsid="$Id: testSTL.cc 6723 2008-04-17 14:18:15Z craftit $"
//! lib=RavlCore
//! file="Ravl/Core/System/testSTL.cc"
//! docentry="Ravl.API.Core.Misc"
//! author="Robert Crida"

#include "Ravl/STL.hh"
#include "Ravl/UnitTest.hh"

using namespace RavlN;

int testVectorStringIO();

int main()
{
  int err;
  if((err = testVectorStringIO()) != 0) {
    cerr << "Test failed line :" << err <<"\n";
    return 1;
  }

  cerr << "STL test passed. \n";
  return 0;
}

int testVectorStringIO() {
  vector<string> strVec;
  strVec.push_back("first");
  strVec.push_back("second");
  strVec.push_back("third");

  vector<string> loadedVec;
  if (!TestBinStreamIO(strVec, loadedVec)) return __LINE__;
  if (strVec != loadedVec) return __LINE__;
  return 0;
}

