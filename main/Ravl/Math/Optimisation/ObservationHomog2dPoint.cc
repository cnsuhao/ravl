// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2002, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"

#include "Ravl/ObservationHomog2dPoint.hh"
#include "Ravl/StateVectorHomog2d.hh"
#include "Ravl/ObsVectorBiGaussian.hh"
#include "Ravl/Vector3d.hh"

namespace RavlN {

  //: Constructor.
  ObservationHomog2dPointBodyC::ObservationHomog2dPointBodyC(
			     const VectorC &nz1, const MatrixRSC &nNi1,
			     const VectorC &nz2, const MatrixRSC &nNi2)
    : ObservationExplicitBodyC(ObsVectorC(nz2,nNi2))
  {
    z1 = nz1;
    Ni1 = nNi1;
  }

  //: Constructor for robust bi-Gaussian observation.
  ObservationHomog2dPointBodyC::ObservationHomog2dPointBodyC(
			     const VectorC &nz1, const MatrixRSC &nNi1,
			     const VectorC &nz2, const MatrixRSC &nNi2,
			     RealT nvar_scale, RealT nchi2_thres)
    : ObservationExplicitBodyC(ObsVectorBiGaussianC(nz2,nNi2,nvar_scale,nchi2_thres))
  {
    z1 = nz1;
    Ni1 = nNi1;
  }

  //: Get point position on first plane
  const VectorC& ObservationHomog2dPointBodyC::GetZ1() const
  {
    return z1;
  }

  //: Get point position inverse covariance on first plane
  const MatrixRSC& ObservationHomog2dPointBodyC::GetNi1() const
  {
    return Ni1;
  }

  //: Evaluate an observation for a single point
  VectorC ObservationHomog2dPointBodyC::EvaluateFunctionH(const StateVectorC &state_vec)
  {
    // we know that the state vector actually represents a 2D homography
    const StateVectorHomog2dC sv(state_vec);
    RavlAssert(sv.IsValid());
    
    const VectorC &z1 = GetZ1(); // 2D point on first plane
    const Matrix3dC &P = sv.GetHomog(); // homography matrix
    RealT zh1 = sv.GetZH1(), zh2 = sv.GetZH2();

    // compute p = P*x1
    Vector3dC p(P[0][0]*z1[0] + P[0][1]*z1[1] + P[0][2]*zh1,
		P[1][0]*z1[0] + P[1][1]*z1[1] + P[1][2]*zh1,
		P[2][0]*z1[0] + P[2][1]*z1[1] + P[2][2]*zh1);

    // evaluate h(x) and store in observation object
    p2[0] = p[0]*zh2/p[2];
    p2[1] = p[1]*zh2/p[2];

    // store computed z-coordinate as well
    z2 = p[2];

    return p2;
  }

  //: Evaluate the Jacobian of an observation for a single point
  MatrixC ObservationHomog2dPointBodyC::EvaluateJacobianHx(const StateVectorC &state_vec)
  {
    // we know that the state vector actually represents a 2D homography
    const StateVectorHomog2dC sv(state_vec);
    RavlAssert(sv.IsValid());
    
    const VectorC &z1 = GetZ1(); // 2D point on first plane
    MatrixC H(2,9); // Jacobian matrix
    RealT zh1 = sv.GetZH1(), zh2 = sv.GetZH2();

    // evaluate Jacobian
    H[0][0] = z1[0]*zh2/z2;     H[1][0] = 0.0;
    H[0][1] = z1[1]*zh2/z2;     H[1][1] = 0.0;
    H[0][2] = zh1*zh2/z2;       H[1][2] = 0.0;
    H[0][3] = 0.0;              H[1][3] = z1[0]*zh2/z2;    
    H[0][4] = 0.0;              H[1][4] = z1[1]*zh2/z2;    
    H[0][5] = 0.0;              H[1][5] = zh1*zh2/z2;    
    H[0][6] = -z1[0]*p2[0]/z2;  H[1][6] = -z1[0]*p2[1]/z2;
    H[0][7] = -z1[1]*p2[0]/z2;  H[1][7] = -z1[1]*p2[1]/z2;
    H[0][8] = -zh1*p2[0]/z2;    H[1][8] = -zh1*p2[1]/z2;
    return H;
  }
}
