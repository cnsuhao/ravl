// This file is used in conjunction with RAVL, Recognition And Vision Library
// Copyright (C) 2007, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU
// General Public License (GPL). See the gpl.licence file for details or
// see http://www.gnu.org/copyleft/gpl.html
// file-header-ends-here
// $Id$
#include "blas2.hh"
#include "blas2_c.hh"

namespace BlasN
{
  void AddOuterProduct(RavlN::MatrixRUTC &M, const RavlN::VectorC &V)
  {
    dsyr_c(M.Size1(), &(M[0][0]), &(V[0]), 1., 1, true, true);
  }
} //end of namespace
