// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlAudioFeatures
//! author="Charles Galambos"
//! docentry="Ravl.Audio.Feature Extraction"
//! userlevel=Normal

#include "Ravl/Option.hh"
#include "Ravl/Audio/FeatureMFCC.hh"
#include "Ravl/DP/SequenceIO.hh"

using namespace RavlN;
using namespace RavlAudioN;

int main(int nargs,char **argv) {
  OptionC opts(nargs,argv);
  StringC inFile = opts.String("","@DEVAUDIO:/dev/dsp","Audio input. ");
  StringC outFile = opts.String("","","Output for features. ");
  IntT sampleRate = opts.Int("sr",16000,"Sample rate. ");
  opts.Check();
  
  DPIPortC<Int16T> inp;
  if(!OpenISequence(inp,inFile)) {
    cerr << "Failed to open input file '" << inFile << "'\n";
    return 1;
  }
  
  DPOPortC<VectorC> outp;
  if(!outFile.IsEmpty()) {
    if(!OpenOSequence(outp,outFile)) {
      cerr << "Failed to open output file '" << outFile << "'\n";
      return 1;
    }
  }
  inp.SetAttr("samplerate",sampleRate);
  FeatureMFCCC fextract(sampleRate);
  fextract.Input() = inp;
  VectorC vec;
  while(1) {
    fextract.Get(vec);
#if 0
    for(int i = 0;i < vec.Size();i++)
      cout << Round(vec[i]) << " ";
    cout << "\n";
#else
    cout << vec << "\n";
#endif
    if(outp.IsValid())
      outp.Put(vec);
  }
  
  return 0;
}
