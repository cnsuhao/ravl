// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_INTC_HEADER
#define RAVL_INTC_HEADER 1
////////////////////////////////////////////////////
//! docentry="Ravl.Core.Misc"
//! userlevel=Normal
//! rcsid="$Id$"
//! file="Ravl/Core/Base/IntC.hh"
//! lib=RavlCore
//! author="Charles Galambos"
//! date="26/10/98"

#include "Ravl/Types.hh"
#include "Ravl/Assert.hh"

//: Ravl namespace.

namespace RavlN {
  
  //: Simple integer.
  // Its main feature is the default constructor sets 
  // its value to zero.  Usefull in things like histograms.
  
  class IntC {
  public:
    IntC() 
      : v(0) {}
    //: Default value, 0.
    
    IntC(IntT nv) 
      : v(nv) {}
    //: Construct an int.
    
    IntC Copy() const { return IntC(v); }
    //: Make a copy.
    
    operator IntT() const { return v; }
    //: Convert to a plain int.
    
    bool operator==(const IntC &oth) const { return v == oth.v; }
    //: Comparison operator.
    
    bool operator==(IntT ov) const { return v == ov; }
    //: Comparison operator.
    
    UIntT Hash() const { return (UIntT) v; }
    //: Hash it.
    
    IntT operator++(int) { return v++; }
    //: Increment.
    
    IntT operator--(int) { return v--; }
    //: Decrement
    
    IntT operator++() { return ++v; }
    //: Increment.
    
    IntT operator--() { return --v; }
    //: Decrement
    
    IntT v;
  };
  
  ostream &operator<<(ostream &out,const IntC &x);  
  istream &operator>>(istream &in,IntC &x);
  
  //: Simple unsigned  integer.
  // Its main feature is the default constructor sets 
  // its value to zero.  Usefull in things like histograms.
  
  class UIntC {
  public:
    UIntC() 
      : v(0) {}
    //: Default value, 0.
    
    UIntC(UIntT nv) 
      : v(nv) {}
    //: Construct an int.
    
    UIntC Copy() const { return UIntC(v); }
    //: Make a copy.
    
    operator UIntT() const { return v; }
    //: Convert to a plain int.
    
    bool operator==(const UIntC &oth) const { return v == oth.v; }
    //: Comparison operator.
    
    bool operator==(UIntT ov) const { return v == ov; }
    //: Comparison operator.
    
    UIntT Hash() const { return v; }
    //: Hash it.
    
    UIntT operator++(int) { return v++; }
    //: Increment.
    
    UIntT operator--(int) { return v--; }
    //: Decrement
    
    UIntT operator++() { return ++v; }
    //: Increment.
    
    UIntT operator--() { return --v; }
    //: Decrement

    UIntT operator-=(const UIntC &oth) {
      RavlAssert(v >= oth.v);
      v -= oth.v;
      return v;
    }
    //: Take another UIntT from this one.
    
    UIntT v;
  };
  
  ostream &operator<<(ostream &out,const UIntC &x);  
  istream &operator>>(istream &in,UIntC &x);
  
}
#endif
