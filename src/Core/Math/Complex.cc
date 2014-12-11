// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001-12, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// Copyright (C) 2000 Daniel A. Atkinson    All rights reserved.
// This code may be redistributed under the terms of the GNU library
// public license (LGPL). ( See the lgpl.license file for details.)
// file-header-ends-here
/////////////////////////////////////////////////////
//! rcsid="$Id$"
//! lib=RavlMath
//! file="Ravl/Core/Math/Complex.cc"

//#include "Ravl/config.h"
#include "Ravl/Complex.hh"
#include "Ravl/Stream.hh"

#if RAVL_HAVE_CSQRT
extern "C" {
  #include <complex.h>
}
#endif


namespace RavlN {
  ostream & 
  operator<<(ostream & outS, const ComplexC & x) {
    outS << x.Re() << ' ' << x.Im();
    return(outS);
  }

  istream & 
  operator>>(istream & inS, ComplexC & x) {
    inS >> x.Re() >> x.Im();
    return(inS);
  }
  
  ComplexC Sqrt(const ComplexC & a) {
#if RAVL_HAVE_CSQRT
    double complex arg;
    arg = a.Re() +1.0I * a.Im();
    double complex root = csqrt(arg);
    return ComplexC (creal(root), cimag(root));
    //: Returns one of the complex square roots of a complex number
    // The 2nd root is the -ve of the given one
#else
    double r;
    r = sqrt(a.Re() * a.Re() + a.Im() * a.Im());
    r = sqrt(ldexp(r + fabs(a.Re()), -1));
    ComplexC ret;
    if (r != 0.)
    {
      if (a.Re() >= 0.)
      {
        ret.Re() = r;
        ret.Im() = ldexp(a.Im() / r, -1);
      }
      else
      {
        ret.Re() = ldexp(fabs(a.Im()) / r, -1);
        if (a.Im() >= 0.)
          ret.Im() = r;
        else
          ret.Im() = -r;
      }
    }
    return ret;
#endif // RAVL_HAVE_CSQRT
  }
}

