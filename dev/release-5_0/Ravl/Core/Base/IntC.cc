// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
////////////////////////////////////////////////////////
//! rcsid="$id$"
//! lib=RavlCore
//! file="Ravl/Core/Base/IntC.cc"

#include "Ravl/IntC.hh"
#include "Ravl/Stream.hh"

namespace RavlN {

#if RAVL_STRICT_NAMESPACES
  //using namespace std;
#endif

  ostream &operator<<(ostream &out,const IntC &x) {
    out << ((int) x.v);
    return out;
  }
  
  istream &operator>>(istream &in,IntC &x) {
    in >> ((int &) x.v);
    return in;
  }

  ostream &operator<<(ostream &out,const UIntC &x) {
    out << ((unsigned int) x.v);
    return out;
  }
  
  istream &operator>>(istream &in,UIntC &x) {
    in >> ((unsigned int &) x.v);
    return in;
  }

}
