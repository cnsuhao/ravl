// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLSIGNAL_REALCEPSTRAL
#define RAVLSIGNAL_REALCEPSTRAL 1
/////////////////////////////////////////////
//! rcsid="$Id$"
//! userlevel=Normal
//! docentry="Ravl.Math.Signals.1D"
//! lib=RavlMath

#include "Ravl/Math.hh"
#include "Ravl/SArray1d.hh"
#include "Ravl/Types.hh"
#include "Ravl/FFT1d.hh"

namespace RavlN {
  
  //: Compute the real cepstral 
  
  class RealCepstralC {
  public:
    RealCepstralC(SizeT size);
    //: Constructor.

    SArray1dC<RealT> Apply(const SArray1dC<RealT> &data);
    //: Compute the real cepstral of data.

  protected:
    FFT1dC fft;
    FFT1dC ifft;
  };


}  


#endif
