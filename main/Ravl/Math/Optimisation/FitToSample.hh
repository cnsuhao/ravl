// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2002, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLMATH_FITTOSAMPLE_HEADER
#define RAVLMATH_FITTOSAMPLE_HEADER 1
//! userlevel=Normal
//! author="Phil McLauchlan"
//! date="24/7/2002"
//! rcsid="$Id$"
//! docentry="Ravl.Math.Optimisation"
//! example="OrthogonalRegressionTest.cc Homography2dFitTest.cc FitQuadraticTest.cc"
//! lib=RavlOptimise

#include "Ravl/RefCounter.hh"
#include "Ravl/StateVector.hh"
#include "Ravl/Observation.hh"
#include "Ravl/DList.hh"

namespace RavlN {
  
  //! userlevel=Develop
  //: Observation sampling base class
  class FitToSampleBodyC
    : public RCBodyVC
  {
  public:
    FitToSampleBodyC()
    {}
    //: Default constructor.
    
    virtual StateVectorC FitModel(DListC<ObservationC> sample);
    //: Fit model parameters to sample of observations
  };

  //! userlevel=Normal
  //! autoLink=on
  //: This class fits a state parameter model to a sample of observations.
  class FitToSampleC
    : public RCHandleC<FitToSampleBodyC>
  {
  public:    
    FitToSampleC()
    {}
    //: Default constructor.
    // Creates an invalid handle.
    
    FitToSampleC(FitToSampleBodyC &bod)
      : RCHandleC<FitToSampleBodyC>(bod)
    {}
    //: Body constructor.

    FitToSampleBodyC &Body()
    { return RCHandleC<FitToSampleBodyC>::Body(); }
    //: Access body.

    const FitToSampleBodyC &Body() const
    { return RCHandleC<FitToSampleBodyC>::Body(); }
    //: Access body.
    
  public:
    StateVectorC FitModel(DListC<ObservationC> sample)
    { return Body().FitModel(sample); }
    //: Fit model parameters to sample of observations
    // Supply this function in a sub-class.
  };
}


#endif
