// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2002, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlMath
//! file="Ravl/Math/LinearAlgebra/FixedSize/testSpecificFMatrix.cc"

#include "Ravl/Matrix3d.hh"
#include "Ravl/Matrix2d.hh"
#include "Ravl/Vector2d.hh"
#include "Ravl/Vector3d.hh"

using namespace RavlN;

int testMatrix2d();
int testMatrix3d();

int main() {
  int ln;
  if((ln = testMatrix2d()) != 0) {
    cerr << "Test failed on line " << ln << "\n";
    return 1;
  }
  if((ln = testMatrix3d()) != 0) {
    cerr << "Test failed on line " << ln << "\n";
    return 1;
  }
  cout << "Test passed ok. \n";
  return 0;
}

int testMatrix2d() {
  Matrix2dC x(1,2,3,4);
  Matrix2dC y;
  if(!x.Invert(y)) {
    cerr << "Failed to invert matrix. \n";
    return __LINE__;
  }
  
  if(((y * x) - Matrix2dC(1,0,0,1)).SumOfSqr() > 0.0000001)
    return __LINE__;

  Matrix2dC z(3,5,5,2);
  Vector2dC v;
  EigenVectors(z,x,v);
  //cout << "x=" << x << " v=" << v << "\n";
  
  Matrix2dC md(v[0],0,0,v[1]);
  Matrix2dC m = x * md * x.Inverse();
  //  cerr << "m=" << m << "\n";
  if((m - z).SumOfSqr() > 0.0000001) return __LINE__;
  
  return 0;
}

int testMatrix3d() {
  Matrix3dC x(1,7,3,4,5,3,7,8,9);
  Matrix3dC y;
  if(!x.Invert(y)) {
    cerr << "Failed to invert matrix. \n";
    return __LINE__;
  }
  
  if(((y * x) - Matrix3dC(1,0,0,0,1,0,0,0,1)).SumOfSqr() > 0.0000001)
    return __LINE__;  

  // Joel's test.
  
  Matrix3dC E(0, -8.15447e-14, -1.22998e-12,
              -3.47383, 1.35606, 11.4019,
              -1.35606, -11.8774, 0.999454);
  Matrix3dC Eu, Ev;
  Vector3dC Ed;
  
  Ed = SVD(E, Eu, Ev);
  
  if((E - (Eu*Matrix3dC(Ed[0],0,0,0,Ed[1],0,0,0,Ed[2])*Ev.T())).SumOfSqr() > 0.00001)
    return __LINE__;
  
  Vector3dC v3a(1,2,3),v3b(3,2,1);
  Vector3dC v = v3a + v3b;
  if(v[0] != 4 || v[1] != 4 || v[2] != 4) return __LINE__;
  return 0;
}
