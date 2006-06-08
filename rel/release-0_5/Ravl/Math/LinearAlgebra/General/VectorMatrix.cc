// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//////////////////////////////////////////////////////////////////
//! author="Kieron Messer"
//! rcsid="$Id$"
//! lib=RavlMath

#include "Ravl/Stream.hh"
#include "Ravl/VectorMatrix.hh"

namespace RavlN {


  VectorMatrixC::VectorMatrixC(const UIntT dim)
    : VectorC(dim),
      MatrixC(dim,dim)
  {}
  
  VectorMatrixC::VectorMatrixC(const VectorC & vec, const MatrixC & mat)
    : VectorC(vec),
      MatrixC(mat)
  {}
  
  VectorMatrixC VectorMatrixC::Copy() const 
  { return VectorMatrixC(VectorC::Copy(),MatrixC::Copy()); }

  void VectorMatrixC::SetZero() {
    VectorC::Fill(0.0);
    MatrixC::Fill(0.0);
  }
  
  void VectorMatrixC::Sort() {
    // The straight insertion method is used for sorting.
    const UIntT vn = Vector().Size();
    
    for (UIntT i=0; i<vn-1; i++) {
      // Find the maximum item in the rest of the vector.
      UIntT   maxIndex = i;
      RealT maxValue = Vector()[i];
      for (UIntT j=i+1; j<vn; j++)
	if (Vector()[j] > maxValue) 
	  maxValue = Vector()[maxIndex=j];
      
      if (maxIndex != i) { 
	// Swap columns of the matrices and items of the vector.
	Vector()[maxIndex]=Vector()[i];
	Vector()[i]=maxValue;
	for (UIntT j=0; j<vn; j++)  { 
	  // Swap two elements of the matrix.
	  RealT value    = Matrix()[j][i];
	  Matrix()[j][i] = Matrix()[j][maxIndex];
	  Matrix()[j][maxIndex]=value;
	}
      }
    }
  }
  
#if 0
  void VectorMatrixC::BubbleSort() {
    // bubble sorting according to vector values
    // the first value of the result vector is the biggest one
    MatrixC & m     = Matrix();
    VectorC & v     = Vector();
    const SizeT dim = v.Size();
    bool  change;
    do {
      change = false;
      for (UIntT i = 0; i < dim-1; ++i) {
	if (v[i] < v[i+1]) {
	  UIntT i1 = i+1;
	  for (UIntT j = 0; j < dim; ++j) {
	    RealT ee = m[j][i];
	    m[j][i] = m[j][i1];
	    m[j][i1] = ee;
	  }
	  RealT ev = v[i];
	  v[i] = v[i1];
	  v[i1] = ev;
	  change = true;
	}
      }
    }while(change);
    return *this;
  }
#endif
  
  //--------------------------------------------------------------------
  // *************** Input - Output ************************************
  //--------------------------------------------------------------------
  
  ostream &operator<<(ostream & s, const VectorMatrixC & vm) {
    const VectorC & v = vm.Vector();
    const MatrixC & m = vm.Matrix();
    s << v << '\n' << m;
    return s;
  }
  
  istream &operator>>(istream & s, VectorMatrixC & vm) {
    VectorC & v = vm.Vector();
    MatrixC & m = vm.Matrix();
    s >> v >> m;
    return s;
  }
  
}