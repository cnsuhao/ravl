// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlMath
//! file="Ravl/Math/Statistics/Histogram/RealHistogram1d.cc"

#include "Ravl/RealHistogram1d.hh"
#include "Ravl/SArr1Iter.hh"

namespace RavlN {
  
  //: Create a histogram.
  
  RealHistogram1dC::RealHistogram1dC(RealT min,RealT max,UIntT steps) {
    RealT range = max - min;
    scale =  range / steps;
    offset = min;
    ((SArray1dC<UIntC> &)(*this)) = SArray1dC<UIntC>(0,steps);
  }
  
  //: Find the total number of votes cast.
  
  UIntT RealHistogram1dC::TotalVotes() const {
    UIntT c = 0;
    for(SArray1dIterC<UIntC> it(*this);it;it++) 
      c += *it;
    return c;
  }
  
  //: Calculate the amount of information represented by the histogram.
  // This is also known as the entropy of the histogram.
  
  RealT RealHistogram1dC::Information() const {
    RealT totalp = 0;
    UIntT total = TotalVotes();
    for(SArray1dIterC<UIntC> it(*this);it;it++) {
      RealT prob = (RealT) *it / total;
      totalp += -prob * Log2(prob);
    }
    return totalp;
  }
  
  //: Calculate the energy represented by the original signal.
  
  RealT RealHistogram1dC::Energy() const {
    UIntT total = TotalVotes();
    RealT sum = 0;
    for(SArray1dIterC<UIntC> it(*this);it;it++)
      sum += Pow((RealT) *it / total,2);
    return sum;
  }

}