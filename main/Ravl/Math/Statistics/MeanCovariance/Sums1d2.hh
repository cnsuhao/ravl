// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2002, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_SUMS1D_HEADER
#define RAVL_SUMS1D_HEADER 1
//! author="Charles Galambos"
//! date="26/8/2002"
//! docentry="Ravl.Math.Statistics"
//! rcsid="$Id$"
//! lib=RavlMath

#include "Ravl/Math.hh"
#include "Ravl/MeanVariance.hh"

namespace RavlN {
  
  //! userlevel=Normal
  //: Sums of a variable.
  // This class provides a way of calculating statistics about
  // a variable. 
  
  class Sums1d2C {
  public:
    Sums1d2C()
      : n(0),
	sum(0),
	sum2(0)
    {}
    //: Default constructor.
    // Sets sums to zero.
    
    Sums1d2C(UIntT nn,RealT nsum,RealT nsum2)
      : n(nn),
	sum(nsum),
	sum2(nsum2)
    {}
    //: Constructor from sum elements.
    
    void operator+=(RealT val) {
      n++;
      sum += val;
      sum2 += Sqr(val);
    }
    //: Add a point.

    void operator-=(RealT val) {
      n--;
      sum -= val;
      sum2 -= Sqr(val);
    }
    //: Remove a point.
    
    void operator+=(const Sums1d2C &s) {
      n += s.n;
      sum += s.sum;
      sum2 += s.sum2;
    }
    //: Add another set of sums.

    void operator-=(const Sums1d2C &s) {
      RavlAssert(s.n < n);
      n += s.n;
      sum -= s.sum;
      sum2 -= s.sum2;
    }
    //: Subtract another set of sums.
    
    UIntT Size() const
    { return n; }
    //: Number of data points.
    
    RealT Sum() const
    { return sum; }
    //: Sum of all data points.

    RealT Sum2() const
    { return sum2; }
    //: Sum of squares of all data points.
    
    MeanVarianceC MeanVariance() const {
      RealT rn = (RealT) n;
      RealT mean  = sum / rn;
      return MeanVarianceC(n,mean,(sum2 - Sqr(sum)/rn)/(rn-1));
    }
    //: Calculate the mean and variance for this sample.

    RealT Variance() const {
      RealT rn = (RealT) n;
      return (sum2 - Sqr(sum)/rn)/(rn-1);
    }
    //: Compute the variance of the sample.
    
    RealT Mean() const {
      RealT rn = (RealT) n;
      return sum / rn;
    }
    //: Compute the mean of the sample.
    
  protected:
    UIntT n;
    RealT sum; // Sum of data.
    RealT sum2; // Sum of square data.
  };

  ostream& operator<<(ostream &s,const Sums1d2C &mv);
  istream& operator>>(istream &s, Sums1d2C &mv);

}


#endif
