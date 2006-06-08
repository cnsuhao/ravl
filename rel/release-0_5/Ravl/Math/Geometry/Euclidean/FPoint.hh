// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLFPOINT_HEADER
#define RAVLFPOINT_HEADER 1
//////////////////////////////////////////////////
//! rcsid=$Id$
//! file="Ravl/Math/Geometry/Euclidean/FPoint.hh"
//! lib=RavlMath
//! userlevel=Develop
//! author="Charles Galambos"
//! date="24/01/2001"
//! docentry="Ravl.Math.Geometry"

#include "Ravl/TFVector.hh"
#include "Ravl/FIndex.hh"

namespace RavlN {

  template <unsigned int N>
  class FVectorC;
  
  //! userlevel=Advanced
  //: Real point with templated size
  
  template<unsigned int N>
  class FPointC
    : public TFVectorC<RealT,N>
  {
  public:
    FPointC()
      {}
    //: Default constructor.
    // The contents of the class are left
    // undefined.
    
    FPointC(const TFVectorC<RealT,N> &oth)
      : TFVectorC<RealT,N>(oth)
      {}
    //: Base constructor.
    
    FPointC(const FIndexC<N> &oth) { 
      for(unsigned int i = 0; i < N;i++) 
	data[i] = (RealT) oth[i];
    }
    //: Construct from an index.
    
    explicit FPointC(const FVectorC<N> &oth)
      : TFVectorC<RealT,N>(oth)
      {}
    //: Vector constructor.
    
    RealT EuclidDistance(const FPointC<N> & i) const 
      { return Sqrt((RealT) SqrEuclidDistance(i)); }
    //: Returns the distance of two indexes in Euclid metric.

  };
}

#endif