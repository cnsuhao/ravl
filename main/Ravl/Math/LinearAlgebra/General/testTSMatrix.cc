// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2002, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlMath
//! file="Ravl/Math/LinearAlgebra/General/testTSMatrix.cc"

#include "Ravl/TSMatrix.hh"
#include "Ravl/TSMatrixFull.hh"
#include "Ravl/TSMatrixPartial.hh"
#include "Ravl/TSMatrixDiagonal.hh"
#include "Ravl/TSMatrixRightUpper.hh"
#include "Ravl/TSMatrixLeftLower.hh"
#include "Ravl/TSMatrixSparse.hh"
#include "Ravl/TSMatrixSymmetric.hh"
#include "Ravl/TSMatrixScaledIdentity.hh"
#include "Ravl/TSMatrixTranspose.hh"
#include "Ravl/Matrix.hh"
#include "Ravl/SMatrix.hh"
#include "Ravl/Vector.hh"

using namespace RavlN;

int testBasic();
int testDiagonal();
int testRightUpper();
int testLeftLower();
int testSparse();
int testSymmetric();

int main() {
  int ln;
  if((ln = testBasic()) != 0) {
    cerr << "Test failed at " << ln << "\n";
    return 1;
  }
  if((ln = testDiagonal()) != 0) {
    cerr << "Test failed at " << ln << "\n";
    return 1;
  }
  if((ln = testRightUpper()) != 0) {
    cerr << "Test failed at " << ln << "\n";
    return 1;
  }
  if((ln = testLeftLower()) != 0) {
    cerr << "Test failed at " << ln << "\n";
    return 1;
  }
  if((ln = testSparse()) != 0) {
    cerr << "Test failed at " << ln << "\n";
    return 1;
  }
  if((ln = testSymmetric()) != 0) {
    cerr << "Test failed at " << ln << "\n";
    return 1;
  }
  cerr << "Test passed. \n";
  return 0;
}


int MatrixTest(SMatrixC mat1,SMatrixC mat2) {
  cerr <<"mat1=" << mat1.TMatrix() << "\n";
  cerr <<"mat2=" << mat2.TMatrix() << "\n";
  cerr << "Check Mul. \n";
  MatrixC gt = mat1.TMatrix() * mat2.TMatrix();
  SMatrixC testR = mat1 * mat2;
  RealT error = MatrixC(testR.TMatrix() - gt).SumOfAbs();
  if(error > 0.000000001) {
    cerr << "gt  =" << gt << "\n";
    cerr << "test=" << testR.TMatrix() << "\n";
    return __LINE__;
  } 
  
  cerr << "Check TMul. \n";
  gt = mat1.TMatrix().T() * mat2.TMatrix();
  testR = mat1.TMul(mat2);
  error = MatrixC(testR.TMatrix() - gt).SumOfAbs();
  if(error > 0.000000001) {
    cerr << "gt  =" << gt << "\n";
    cerr << "test=" << testR.TMatrix() << "\n";
    return __LINE__;
  }
  
  cerr << "Check MulT. \n";
  gt = mat1.TMatrix() * mat2.TMatrix().T();
  testR = mat1.MulT(mat2);
  error = MatrixC(testR.TMatrix() - gt).SumOfAbs();
  if(error > 0.000000001) return __LINE__;

  cerr << "Check ATA. \n";
  gt = mat1.TMatrix().ATA();
  testR = mat1.ATA();
  error = MatrixC(testR.TMatrix() - gt).SumOfAbs();
  if(error > 0.000000001) {
    cerr << "gt  =" << gt << "\n";
    cerr << "test=" << testR.TMatrix() << "\n";
    return __LINE__;
  }

  cerr << "Check AAT. \n";
  gt = mat1.TMatrix().AAT();
  testR = mat1.AAT();
  error = MatrixC(testR.TMatrix() - gt).SumOfAbs();
  if(error > 0.000000001) {
    cerr << "gt  =" << gt << "\n";
    cerr << "test=" << testR.TMatrix() << "\n";
    return __LINE__;
  }
  
  cerr << "Check Mul(VectorC)\n";
  VectorC vec = RandomVector(mat1.Cols());
  VectorC tres = mat1.Mul(vec);
  VectorC vgt = mat1.TMatrix() * vec;
  error = VectorC(tres - vgt).SumOfAbs();
  if(error > 0.000000001) {
    cerr << "Error=" << error << "\n";
    cerr << "Res=" << tres << "\n";
    cerr << "gt =" << vgt << "\n";
    return __LINE__;
  }
  
  cerr << "Check TMul(VectorC)\n";
  vec = RandomVector(mat1.Rows());
  tres = mat1.TMul(vec);
  vgt = mat1.TMatrix().TMul(vec);
  error = VectorC(tres - vgt).SumOfAbs();
  if(error > 0.000000001) {
    cerr << "Error=" << error << "\n";
    cerr << "Res=" << tres << "\n";
    cerr << "gt =" << vgt << "\n";
    return __LINE__;
  }
  
  return 0;
}


int testBasic() {
  cerr << "testBasic(). \n";
  int ln;
  MatrixC fm1(10,10);
  RealT v = 1;
  for(SArray2dIterC<RealT> it(fm1);it;it++)
    *it = v++;
  MatrixC fm2(10,10);
  for(SArray2dIterC<RealT> it(fm2);it;it++)
    *it = v++;
  TSMatrixFullC<RealT> rtsm1(fm1);  
  TSMatrixFullC<RealT> rtsm2(fm2);
  if(rtsm1.Rows() != 10) return __LINE__;
  if(rtsm1.Cols() != 10) return __LINE__;
  
  if((ln = MatrixTest(rtsm1,rtsm2)) != 0) {
    cerr<< "MatrixTest() failed on line " << ln << "\n";
    return __LINE__;
  }  
  return 0;
}

int testDiagonal() {
  cerr << "testDiagonal(). \n";
  TSMatrixDiagonalC<RealT> t1(RandomVector(5));
  TSMatrixDiagonalC<RealT> t2(RandomVector(5));
  int ln;
  if((ln = MatrixTest(t1,t2)) != 0) {
    cerr<< "MatrixTest() failed on line " << ln << "\n";
    return __LINE__;
  }  
  return 0;
}

int testRightUpper() {
  cerr << "testRightUpper() \n";
  TSMatrixRightUpperC<RealT> mru1(2);
  RealT c = 2;
  for(UIntT i = 0;i < mru1.Rows();i++)
    for(UIntT j = i;j < mru1.Cols();j++)
      mru1.Element(i,j,c++);
  //cerr <<"Mat=" << mru.TMatrix() << "\n"; 
  TSMatrixRightUpperC<RealT> mru2(2);
  for(UIntT i = 0;i < mru2.Rows();i++)
    for(UIntT j = i;j < mru2.Cols();j++)
      mru2.Element(i,j,c++);
  int ln;
  if((ln = MatrixTest(mru1,mru2)) != 0) {
    cerr<< "MatrixTest() failed on line " << ln << "\n";
    return __LINE__;
  }  
  return 0;
}

int testLeftLower() {
  cerr << "testLeftLower() \n";
  TSMatrixLeftLowerC<RealT> mru1(4);
  RealT c = 2;
  for(UIntT i = 0;i < mru1.Rows();i++)
    for(UIntT j = 0;j <= i;j++)
      mru1.Element(i,j,c++);
  //cerr <<"Mat=" << mru.TMatrix() << "\n"; 
  TSMatrixLeftLowerC<RealT> mru2(4);
  for(UIntT i = 0;i < mru1.Rows();i++)
    for(UIntT j = 0;j <= i;j++)
      mru2.Element(i,j,c++);
  int ln;
  if((ln = MatrixTest(mru1,mru2)) != 0) {
    cerr<< "MatrixTest() failed on line " << ln << "\n";
    return __LINE__;
  }  
  return 0;
}

int testSparse() {
  cerr << "testSparse() \n";
  TSMatrixSparseC<RealT> sm1(4,4);
  sm1.Element(1,2,1);
  sm1.Element(0,3,2);
  sm1.Element(2,2,3);
  sm1.Element(3,1,3);
  sm1.Element(0,0,4);
  sm1.Element(2,0,5);
  
  TSMatrixSparseC<RealT> sm2(4,4);
  sm2.Element(2,0,1);
  sm2.Element(0,2,2);
  sm2.Element(1,3,3);
  sm2.Element(2,2,4);
  sm2.Element(3,3,5);
  int ln;
  if((ln = MatrixTest(sm1,sm2)) != 0) {
    cerr<< "MatrixTest() failed on line " << ln << "\n";
    return __LINE__;
  }
  
  return 0;
}

int testSymmetric() {
  cerr << "testSummetric() \n";
  MatrixC mat = RandomPositiveDefiniteMatrix(4);
  TSMatrixSymmetricC<RealT> sym(mat);
  TSMatrixLeftLowerC<RealT> ru = sym.Cholesky();
  //cerr << "RU=" << ru.TMatrix() << "\n";
  TSMatrixC<RealT> res = ru.MulT(ru);
  RealT err = (res - sym).SumOfAbs();
  if(err > 0.0000001) return __LINE__;
  
  return 0;
}
