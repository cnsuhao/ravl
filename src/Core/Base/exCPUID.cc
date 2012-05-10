// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006-12, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#include "Ravl/CPUID.hh"
#include "Ravl/Option.hh"
#include <iostream>
//! lib=RavlCore


using namespace RavlN;


int main(int argc, char **argv) {

  CPUIDC cpuid;

  cpuid.Info();

  //test non class functions
  cout << "Non class functions test\n";
  cout << "MMX: " << MMX()  << endl;
  cout << "SSE: " << SSE()  << endl;
  cout << "SSE2:" << SSE2() << endl;
  cout << "HTT: " << HTT()  << endl;
}

