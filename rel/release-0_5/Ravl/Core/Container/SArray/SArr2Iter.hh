// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLSARR2ITER_HEADER
#define RAVLSARR2ITER_HEADER 1
////////////////////////////////////////////////////////////
//! docentry="Ravl.Core.Arrays.2D"
//! rcsid="$Id$
//! file="Ravl/Core/Container/SArray/SArr2Iter.hh"
//! lib=RavlCore
//! author="Charles Galambos"
//! date="10/09/98"
//! userlevel=Advanced

#include "Ravl/SArray2d.hh"
#include "Ravl/SArr1Iter.hh"
#include "Ravl/BfAcc2Iter.hh"
#include "Ravl/Index2d.hh"

namespace RavlN {

  //! userlevel=Advanced
  //: SArray2dC iterator.
  // Simple 2d array iterator.
  
  template<class DataT>
  class SArray2dIterC 
  : public BufferAccess2dIterC<DataT>
  {
  public:
    SArray2dIterC()
      {}
    //: Default constructor.
    
    SArray2dIterC(const SArray2dC<DataT> &narr)
      : arr(narr)
      { First(); }
    //: Constructor.
    
    const SArray2dIterC<DataT> &operator=(SArray2dC<DataT> &narr) {
      arr = narr;
      First();
      return *this;
    }
    //: Assignment to an array.
    
    inline void First()
      { BufferAccess2dIterC<DataT>::First(arr,arr.Size2()); }
    //: Goto first element in array.
    
    Index2dC Index() const {
      return BufferAccess2dIterC<DataT>::Index(arr.ReferenceElm());
    }
    //: Get current index.
    // This is a little slow.
    
  protected:
    SArray2dC<DataT> arr;
  };
  
  
}

#endif