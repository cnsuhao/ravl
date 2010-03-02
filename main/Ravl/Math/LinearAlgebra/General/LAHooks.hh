/*
 * LAHooks.hh
 *
 *  Created on: 2 Mar 2010
 *      Author: Alexey Kostin
 */

#ifndef LAHOOKS_HH_
#define LAHOOKS_HH_

namespace RavlN {
  class VectorC;
  class MatrixC;

  extern bool (*g_EigenVectorsSymmetric)(VectorC &resVals, MatrixC &resVecs, const MatrixC &M);

  inline bool EigenVectorsSymmetric(VectorC &resVals, MatrixC &resVecs, const MatrixC &M)
  { return (*g_EigenVectorsSymmetric)(resVals, resVecs, M); }
}

#endif /* LAHOOKS_HH_ */
