// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlCore
//! file="Ravl/Core/Base/IndexRange2d.cc"

#include "Ravl/IndexRange2d.hh"
#include "Ravl/BinStream.hh"

namespace RavlN {
  
  IndexRange2dC::IndexRange2dC(const Index2dC &center,SizeT nrows,SizeT ncols) {
    if(nrows > 0)
      rows = IndexRangeC(center.Row() - (nrows-1)/2,center.Row() + nrows/2);
    else
      rows = IndexRangeC(center.Row(),center.Row());
    if(ncols > 0)
      cols = IndexRangeC(center.Col() - (ncols-1)/2,center.Col() + ncols/2);
    else
      cols = IndexRangeC(center.Col(),center.Col());
  }
  

  //: Rotate rectangle 180 degree's around the given center.
  
  IndexRange2dC IndexRange2dC::Rotate180(Index2dC centre) {
    return IndexRange2dC((End() * -1) + (centre*2),(Origin() * -1) + (centre * 2));
  }

  ostream &operator<<(ostream &s,const IndexRange2dC &ir) {
    s << ir.Range1() << ' ' << ir.Range2();
    return s;
  }
  
  istream &operator>>(istream &s,IndexRange2dC &ir) {
    s >> ir.Range1() >> ir.Range2();
    return s;
  }
  
  BinOStreamC &operator<<(BinOStreamC &s,const IndexRange2dC &ir) {
    s << ir.Range1() << ir.Range2();
    return s;
  }
  
  BinIStreamC &operator>>(BinIStreamC &s,IndexRange2dC &ir) {
    s >> ir.Range1() >> ir.Range2();
    return s;
  }
  
}
