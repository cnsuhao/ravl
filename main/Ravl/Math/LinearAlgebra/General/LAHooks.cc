/*
 * LAHooks.cc
 *
 *  Created on: 2 Mar 2010
 *      Author: Alexey Kostin
 */

#include "Ravl/LAHooks.hh"

#include "Ravl/Vector.hh"
#include "Ravl/Matrix.hh"
#include "Ravl/VectorMatrix.hh"
#include "Ravl/Eigen.hh"


namespace RavlN {

static bool EigenVectorsSymmetric_default(VectorC &resVals, MatrixC &resVecs, const MatrixC &mat) {
  EigenValueC<RealT> ev(mat);
  resVals = ev.EigenValues();
  resVecs = ev.EigenVectors();
  return true;
}

bool (*g_EigenVectorsSymmetric)(VectorC &resVals, MatrixC &resVecs, const MatrixC &M) = &EigenVectorsSymmetric_default;

}
