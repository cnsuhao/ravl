// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlDevAudio
//! author="Charles Galambos"
//! docentry="Ravl.Contrib.Audio IO"

#include "Ravl/IO.hh"
#include "Ravl/Option.hh"
#include "Ravl/DP/SequenceIO.hh"

using namespace RavlN;

int main(int nargs,char **argv) {
  OptionC opt(nargs,argv);
  StringC dev = opt.String("i","@DEVAUDIO:/dev/audio","Sound device.");
  opt.Check();
  
  SArray1dC<Int16T> buff(16000);
  {
    DPIPortC<Int16T> in;
    if(!OpenISequence(in,dev)) {
      cerr << "Failed to open input : " << dev << "\n";
      return 1;
    }
    if(!in.GetArray(buff)) {
      cerr << "Failed to read array of data. \n";
    }
  }
  
  {
    DPOPortC<Int16T> out;
    if(!OpenOSequence(out,dev)) {
      cerr << "Failed to open output : " << dev << "\n";
      return 1;
    }
    if(!out.PutArray(buff)) {
      cerr << "Failed to write array of data. \n";
    }
  }
  
  return 0;
}
