// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2002, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLMATH_POINT2DPAIR_HEADER
#define RAVLMATH_POINT2DPAIR_HEADER 1
//! userlevel=Normal
//! author="Phil McLauchlan"
//! date="24/7/2002"
//! rcsid="$Id$"
//! docentry="Ravl.Math.Optimisation.Examples"
//! example="Homography2dFitTest.cc"
//! lib=RavlOptimise

#include "Ravl/RefCounter.hh"
#include "Ravl/Vector2d.hh"

namespace RavlN {

  //! userlevel=Develop
  //: 2D point pair body
  
  class Point2dPairBodyC
    : public RCBodyVC
  {
  public:
    Point2dPairBodyC(const Vector2dC &nz1, const MatrixRSC &nNi1,
		     const Vector2dC &nz2, const MatrixRSC &nNi2)
      : rz1(nz1),
	rz2(nz2),
	rNi1(nNi1),
	rNi2(nNi2)
    {}
    //: Constructor.

    const Vector2dC & z1() const
    {
      return rz1;
    }
    //: Get first point.

    const Vector2dC & z2() const
    {
      return rz2;
    }
    //: Get second point.

    const MatrixRSC & Ni1() const
    {
      return rNi1;
    }
    //: Get first point inverse covariance.

    const MatrixRSC & Ni2() const
    {
      return rNi2;
    }
    //: Get second point inverse covariance.

  private:
    const Vector2dC rz1, rz2;
    const MatrixRSC rNi1, rNi2;
  };

  //! userlevel=Normal
  //! autoLink=on
  //: 2D point pair class
  class Point2dPairC
    : public RCHandleC<Point2dPairBodyC>
  {
  public:
    Point2dPairC(const Vector2dC &z1, const MatrixRSC &Ni1,
		 const Vector2dC &z2, const MatrixRSC &Ni2)
      : RCHandleC<Point2dPairBodyC>(*new Point2dPairBodyC(z1,Ni1,z2,Ni2))
    {}
    //: Constructor
    // This sticks the individual point observation vectors and inverse
    // covariance matrices together.

  public:
    Point2dPairC(Point2dPairBodyC &bod)
      : RCHandleC<Point2dPairBodyC>(bod)
    {}
    //: Body constructor.
    
    Point2dPairBodyC &Body()
    { return RCHandleC<Point2dPairBodyC>::Body(); }
    //: Access body.

    const Point2dPairBodyC &Body() const
    { return RCHandleC<Point2dPairBodyC>::Body(); }
    //: Access body.
    
  public:
    const Vector2dC &z1() const
    { return Body().z1(); }
    //: Get first point.

    const Vector2dC &z2() const
    { return Body().z2(); }
    //: Get second point.

    const MatrixRSC &Ni1() const
    { return Body().Ni1(); }
    //: Get first point inverse covariance.

    const MatrixRSC &Ni2() const
    { return Body().Ni2(); }
    //: Get second point inverse covariance.
  };
}  


#endif
