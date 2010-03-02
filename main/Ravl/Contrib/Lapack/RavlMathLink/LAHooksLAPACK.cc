/*
 * LAHooksLAPACK.cc
 *
 *  Created on: 2 Mar 2010
 *      Author: Alexey Kostin
 */

#include "Ravl/LAHooks.hh"

#include "Ravl/Vector.hh"
#include "Ravl/Matrix.hh"
#include "Ravl/Lapack/ev_c.hh"

#define DODEBUG 1
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlN {

//! compute eigen values and eigen vectors of symmetric matrix
static bool EigenVectorsSymmetric_LAPACK(VectorC &resVals, MatrixC &resVecs, const MatrixC &mat) {
  const SizeT mSize = mat.Size1();
  resVals = VectorC(mSize);
  resVecs = mat.Copy();
  dsyev_c(mSize, &(resVecs[0][0]), &(resVals[0]), true, true, true, false);
  return true;
}


int LAHooksLAPACKInit() {
  ONDEBUG(cerr << "using LAPACK\n");
  g_EigenVectorsSymmetric = &EigenVectorsSymmetric_LAPACK;
  return 0;
}

static int a = LAHooksLAPACKInit();

}
