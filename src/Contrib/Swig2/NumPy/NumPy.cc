// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2011, OmniPerception Ltd and University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! lib=RavlIntelMKL

#include "Ravl/String.hh"
#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL RAVL_ARRAY_API
#include "numpy/arrayobject.h"

namespace RavlN {
  int InitNumPy()
  {
    std::cerr << "using NumPy\n";
    return _import_array();
  }
  static int a = InitNumPy();
}

