// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLScalMath_HH
#define RAVLScalMath_HH
//////////////////////////////////////////////////////////
//! file="Ravl/Core/Math/ScalMath.hh"
//! lib=RavlCore
//! userlevel=Normal
//! author="Radek Marik"
//! docentry="Ravl.Math"
//! rcsid="$Id$"
//! date="02/11/95"

#include "Ravl/Types.hh"
#include "Ravl/StdMath.hh"
#include "Ravl/Assert.hh"

namespace RavlN {
  
#if RAVL_NEW_ANSI_CXX_DRAFT
  static const IntT RavlPreComputedFactorialSize = 32; 
  // The number of precomputed factorials.
  
  extern RealT RavlPreComputedFactorial[RavlPreComputedFactorialSize]; 
  // The table of precomputed factorials.   
#else
  extern const IntT RavlPreComputedFactorialSize; 
  // The number of precomputed factorials.
  
  extern RealT RavlPreComputedFactorial[32]; // NB. !!! Size must be same as 'factorialSize' !!!
  // The table of precomputed factorials.     
#endif
  
  //! userlevel=Normal
  
  inline IntT Factorial(UIntT n) {
    IntT fac = 1;
    for (; n > 1; n--)
      fac *= n;
    return fac;
  }
  // Returns the factorial of the integer 'n'. The result is computed
  // using integer arithmetic.
  
  inline RealT RBinomCoeff(IntT n, IntT k) {
    double numerator = 1.0;
    for (IntT i = n; i > n-k; i--)
      numerator *= i;
    return numerator/Factorial(k);
  }
  // Returns the binomial coefficient (n over k) as a real number.
  
    
}
#endif
