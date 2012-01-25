// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2002, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLMATH_FITHOMOG2DPOINTS_HEADER
#define RAVLMATH_FITHOMOG2DPOINTS_HEADER 1
//! userlevel=Normal
//! author="Phil McLauchlan"
//! date="24/7/2002"
//! rcsid="$Id$"
//! docentry="Ravl.API.Pattern Recognition.Optimisation2.Cases"
//! example="Homography2dFitTest.cc"
//! lib=RavlOptimise
//! file="Ravl/Math/Optimisation/FitHomog2dPoints.hh"

#include "Ravl/FitToSample.hh"
#include "Ravl/Projection2d.hh"
#include "Ravl/SArray1d.hh"

namespace RavlN {
  
  class Point2dPairObsC;

  //! userlevel=Develop
  //: Body class for fitting a 2D homography to a sample of 2D points
  class FitHomog2dPointsBodyC
    : public FitToSampleBodyC
  {
  public:
    FitHomog2dPointsBodyC(RealT zh1, RealT zh2);
    //: Constructor for a class to fit a 2D homography to pairs of points

    FitHomog2dPointsBodyC();
    //: Constructor for a class to fit a 2D homography to pairs of points
    
    virtual StateVectorC FitModel(DListC<ObservationC> sample);
    //: Fit 2D homography to sample of 2D point observations using least mean squares

    virtual Projection2dC FitModelRobust
    ( const DListC<ObservationC> obsList,
      UIntT noRansacIterations=100,
      RealT ransacChi2Thres=1.0,
      RealT compatChi2Thres=2.0,
      UIntT noLevMarqIterations=4,
      RealT lambdaStart=0.01,
      RealT lambdaFactor=0.1 );
    //: Fit model parameters to sample of observations using robust method

    virtual SArray1dC<bool> Compatibility();
    //: Returns array of booleans to indicate sample compatibility.
    // The array indicates which of the samples used in the FitModelRobust() method was compatible with the fit.<br>
    // The method is only valid after the  FitModelRobust() method is called.
  private:
    RealT zh1, zh2; // 3rd homogeneous coordinates of planes on which points lie
    DListC<bool> inliers;
  };

  //! userlevel=Normal
  //! autoLink=on
  //: This class fits a 2D homography to a sample of 2D points
  class FitHomog2dPointsC
    : public FitToSampleC
  {
  public:
    FitHomog2dPointsC(RealT zh1, RealT zh2)
      : FitToSampleC(*new FitHomog2dPointsBodyC(zh1,zh2))
    {}
    //: Constructor for a class to fit a 2D homography to pairs of points
    // zh1, zh2 are the 3rd homogeneous coordinates of the two planes on which
    // the point pairs lie.

    FitHomog2dPointsC()
      : FitToSampleC(*new FitHomog2dPointsBodyC())
    {}
    //: Constructor for a class to fit a 2D homography to pairs of points
    // The 3rd homogeneous coordinates of the two planes on which
    // the point pairs lie are set to 1.0.
    
    FitHomog2dPointsC(const FitToSampleC &fitter)
      : FitToSampleC(dynamic_cast<const FitHomog2dPointsBodyC *>(BodyPtr(fitter)))
    {}
    //: Base class constructor.
    
    FitHomog2dPointsC(FitHomog2dPointsBodyC &bod)
      : FitToSampleC(bod)
    {}
    //: Body constructor.
    
    FitHomog2dPointsBodyC &Body()
    { return static_cast<FitHomog2dPointsBodyC &>(FitToSampleC::Body()); }
    //: Access body.

    const FitHomog2dPointsBodyC &Body() const
    { return static_cast<const FitHomog2dPointsBodyC &>(FitToSampleC::Body()); }
    //: Access body.

    Projection2dC FitModelRobust
    ( const DListC<ObservationC> obsList,
      UIntT noRansacIterations=100,
      RealT ransacChi2Thres=1.0,
      RealT compatChi2Thres=2.0,
      UIntT noLevMarqIterations=4,
      RealT lambdaStart=0.01,
      RealT lambdaFactor=0.1 )
    { return Body().FitModelRobust(
               obsList, noRansacIterations,
               ransacChi2Thres, compatChi2Thres, noLevMarqIterations,
               lambdaStart, lambdaFactor); }
    //: Fit model parameters to sample of observations using robust method
    // Uses RANSAC and Levenberg-Marquart to create robust fit.

    SArray1dC<bool> Compatibility()
    { return Body().Compatibility(); }
    //: Returns array of booleans to indicate sample compatibility.
    // The array indicates which of the samples used in the FitModelRobust() method was compatible with the fit.<br>
    // The method is only valid after the  FitModelRobust() method is called.

  };
}


#endif
