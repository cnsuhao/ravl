// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_RBFACC2D_HEADER
#define RAVL_RBFACC2D_HEADER 1
///////////////////////////////////////////////////////////
//! rcsid="$Id$"
//! file="Ravl/Core/Container/Buffer/RBfAcc2d.hh"
//! lib=RavlCore
//! userlevel=Normal
//! author="Charles Galambos"
//! date="24/01/2001"
//! docentry="Ravl.Core.Arrays.Buffer"

#include "Ravl/RBfAcc.hh"
#include "Ravl/BfAcc2Iter.hh"
#include "Ravl/Index2d.hh"
#include "Ravl/IndexRange2d.hh"
#include "Ravl/Assert.hh"

namespace RavlN {
  
  class BinOStreamC;
  class BinIStreamC;
  
  //! userlevel=Develop
  //: Basic access to buffer with limited size
  // The class RangeBufferAccessC enables to random indexed access to
  // a sequentially organised continous part of memory called buffer.
  // The access functions check if an accessed element is valid only in
  // debug mode.
  
  template <class DataT>
  class RangeBufferAccess2dC 
    : public RangeBufferAccessC<BufferAccessC<DataT> >
  {
  public:
    RangeBufferAccess2dC(const IndexRangeC &nrng2)
      : rng2(nrng2)
      {}
    //: Constructor.
    
    RangeBufferAccess2dC(const RangeBufferAccessC<BufferAccessC<DataT> > &ab,const IndexRangeC &nrng2)
      : RangeBufferAccessC<BufferAccessC<DataT> >(ab),
      rng2(nrng2)
      {}
    //: Constructor.

    RangeBufferAccess2dC(const RangeBufferAccessC<BufferAccessC<DataT> > &ab,const IndexRange2dC &rect)
      : RangeBufferAccessC<BufferAccessC<DataT> >(ab,rect.Range1()),
      rng2(rect.Range2())
      {}
    //: Construct a access to a rectangle within 'ab' with indexs 'rect'.
    
    RangeBufferAccess2dC(const RangeBufferAccessC<BufferAccessC<DataT> > &ab,const IndexRangeC &r1,const IndexRangeC &r2)
      : RangeBufferAccessC<BufferAccessC<DataT> >(ab,r1),
      rng2(r2)
      {}
    //: Construct a access to a rectangle within 'ab' with ranges r1 and r2.
    
    RangeBufferAccess2dC(const BufferAccessC<BufferAccessC<DataT> > &ab,const IndexRangeC &r1,const IndexRangeC &r2)
      : RangeBufferAccessC<BufferAccessC<DataT> >(r1,ab),
      rng2(r2)
      {}
    //: Construct a access to a rectangle within 'ab' with ranges r1 and r2.
    // All the offsets for the buffers should already be setup.
    
    RangeBufferAccess2dC()
      : rng2(0,-1)
      {}
    //: Default constructor.
    
    inline DataT & operator[](const Index2dC & i) { 
      RavlAssertMsg(rng2.Contains(i.Col()),"Index 1 access out of range. ");
      return RangeBufferAccessC<BufferAccessC<DataT> >::operator[](i.Row())[i.Col()]; 
    }
    //: access to the item array[(i)]
    
    inline const DataT & operator[](const Index2dC & i) const { 
      RavlAssertMsg(rng2.Contains(i.Col()),"Index 1 access out of range. ");
      return RangeBufferAccessC<BufferAccessC<DataT> >::operator[](i.Row())[i.Col()]; 
    }
    //: return the item array[(i)]
    
    inline RangeBufferAccessC<DataT> operator[](IndexC i)
      { return RangeBufferAccessC<DataT>(rng2,RangeBufferAccessC<BufferAccessC<DataT> >::operator[](i)); }
    //: access to the item array[(i)]
    
    inline const RangeBufferAccessC<DataT> operator[](IndexC i) const
      { return RangeBufferAccessC<DataT>(rng2,RangeBufferAccessC<BufferAccessC<DataT> >::operator[](i)); }
    //: return the item array[(i)]
    
    inline const IndexRangeC &Range1() const
      { return Range(); }
    //: Range of first index.
    
    inline const IndexRangeC &Range2() const
      { return rng2; }
    //: Range of second index.

    UIntT Size() const 
    { return Range1().Size() * Range2().Size(); }
    //: Get the total number of elements in the array.
    
    void Fill(const DataT &d);
    //: Fill array with value.
    
    IndexRange2dC Rectangle() const
      { return IndexRange2dC(Range1(),Range2()); }
    //: Return ranges of indexes
    
    IndexRange2dC Frame() const
      { return IndexRange2dC(Range1(),Range2()); }
    //: Return ranges of indexes

    inline bool Contains(const Index2dC & i) const
      { return Range1().Contains(i.Row()) && Range2().Contains(i.Col()); }
    //: Returns TRUE if there is an item of the 2D array
    
    IntT Stride() const {
      if(Range1().Size() < 1)
	return (IntT) rng2.Size();
      return (IntT) (RangeBufferAccessC<BufferAccessC<DataT> >::operator[](IMin()+1).ReferenceElm() -  
		     RangeBufferAccessC<BufferAccessC<DataT> >::operator[](IMin()).ReferenceElm());
    }
    //: Get the stide of the 2d array. 
    
    bool IsContinuous() const 
    { return Stride() == (IntT) rng2.Size(); }
    //: Test if the array is allocated in a continous area of memory.
    // Note: this only checks the first two rows follow each other in
    // memory, this may miss other discontunities.

    bool IsEmpty() const
      { return (Range1().Size() <= 0) || (Range2().Size() <= 0); }
    //: Is rectangle empty ?
    
    BufferAccess2dIterC<DataT> Iter()
      { return BufferAccess2dIterC<DataT>(*this,rng2); }
    //: Creat an iterator for this buffer.
    
  protected:
    IndexRangeC rng2;
  };
  
  template<class DataT>
  void RangeBufferAccess2dC<DataT>::Fill(const DataT &d) {
    for(BufferAccess2dIterC<DataT> it(*this,rng2);it;it++)
      *it = d;
  }
  
  template <class DataT>
  ostream & operator<<(ostream & s, const RangeBufferAccess2dC<DataT> & arr) {
    for(BufferAccess2dIterC<DataT> it(arr,arr.Range2());it;) {
      s << *it;
      for(;it.Next();) 
	s << ' ' << *it;
      s << '\n';
    }    
    return s;
  }
  
  template <class DataT>
  istream & operator>>(istream & s, RangeBufferAccess2dC<DataT> & arr) {
    for(BufferAccess2dIterC<DataT> it(arr,arr.Range2());it;it++) 
      s >> *it;
    return s;
  }

  template<class DataT>
  BinOStreamC &operator<<(BinOStreamC & s, const RangeBufferAccess2dC<DataT> & arr) {
    for(BufferAccess2dIterC<DataT> it(arr,arr.Range2());it;it++)
      s << *it;
    return s;
  }
  
  template<class DataT>
  BinIStreamC &operator>>(BinIStreamC & s, RangeBufferAccess2dC<DataT> & arr) {
    for(BufferAccess2dIterC<DataT> it(arr,arr.Range2());it;it++)
      s >> *it;
    return s;
  }
}

#endif