// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2002, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"

#include "StateVectorLine2d.hh"

namespace RavlN {

  //: Constructor
  StateVectorLine2dBodyC::StateVectorLine2dBodyC(RealT lx, RealT ly, RealT lz,
						 RealT nzh, RealT gauge_weight)
    : StateVectorBodyC(VectorC(lx,ly,lz))
  {
    zh = nzh;
    gauge_weight = gauge_weight;
  }

  //: Constructor
  StateVectorLine2dBodyC::StateVectorLine2dBodyC(RealT lx, RealT ly, RealT lz,
						 RealT nzh, RealT gauge_weight,
						 const VectorC &xstep)
    : StateVectorBodyC(VectorC(lx,ly,lz),xstep)
  {
    zh = nzh;
    gauge_weight = gauge_weight;
  }

  //: Increment the linear system
  // This assumes that the sum of squares of line gradients are normalised
  // to one, so there is nothing to add to vector a.
  bool StateVectorLine2dBodyC::IncrementLS(MatrixRSC &A, VectorC &a)
  {
    A[0][0] += gauge_weight*GetLx()*GetLx();
    A[0][1] += gauge_weight*GetLx()*GetLy();
    A[1][0] += gauge_weight*GetLx()*GetLy();
    A[1][1] += gauge_weight*GetLy()*GetLy();
    return true;
  }

  //: Postprocess 2D line state vector after an adjustment to the parameters.
  bool StateVectorLine2dBodyC::Postprocess()
  {
    RealT lx = GetLx(), ly = GetLy();

    // normalise line gradients to unit norm
    x /= sqrt(lx*lx + ly*ly);

    return true;
  }

  //: Get parameter lx of 2D line lx*xc + ly*yc + lz*zh = 0
  RealT StateVectorLine2dBodyC::GetLx() const
  {
    return x[0];
  }
  
  //: Get parameter ly of 2D line lx*xc + ly*yc + lz*zh = 0
  RealT StateVectorLine2dBodyC::GetLy() const
  {
    return x[1];
  }
  
  //: Get parameter lz of 2D line lx*xc + ly*yc + lz*zh = 0
  RealT StateVectorLine2dBodyC::GetLz() const
  {
    return x[2];
  }

  //: Get 3rd homogeneous plane coordinate
  RealT StateVectorLine2dBodyC::GetZH() const
  {
    return zh;
  }
}
