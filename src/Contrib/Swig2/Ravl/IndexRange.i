// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html

%include "Ravl/Swig2/Types.i"
%include "Ravl/Swig2/Index.i"
%{
#ifdef SWIGPERL
#undef Copy
#endif

#include "Ravl/IndexRange1d.hh"
#include "Ravl/StrStream.hh"


#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN {
  class IndexRangeC {

  public:
    IndexRangeC(size_t dim = 0);
    //: Creates the index range <0, dim-1>.
    
    IndexRangeC(IndexC dim);
    //: Creates the index range <0, dim-1>.

    IndexRangeC(SizeC dim);
    //: Creates the index range <0, dim-1>.

    IndexRangeC(IndexC minIndex, IndexC maxIndex);
    //: Creates the index range <minIndex, maxIndex>.
    
    //:---------------------------------
    //: Access to the object information.
    
    IntT Size() const;
    //: Returns the number of elements in the range.
    
    const IndexRangeC & Range() const;
    //: Returns this object.
    
    const IndexC & Min()  const;
    //: Returns the minimum index of the range.
    
    const IndexC & Max()  const;
    //: Returns the maximum index of the range.
    
    IndexC & Min();
    //: Returns the minimum index of the range.
    
    IndexC & Max();
    //: Returns the maximum index of the range.
    
    IndexC Center() const;
    //: Returns the index in the middle of the range, eg. (Max()+Min()+1)/2.
    
    IndexC CenterD() const;
    //: Returns the index previous the middle of the range, eg. (Max()+Min())/2.
    
    IndexC Percentage(const RealT p) const;
    //: Returns the index which is in the 'p' % of the whole range.
    
    //:-------------------
    //: Logical operations.
    
    bool IsEmpty() const;
    //: Returns true if the minimum limit is bigger than the maximum limit. 
    // The range is empty if Min() > (Max()+1) since zero length ranges are supported
    // Note: if Min() = Max() the range has length 1

    bool IsValid() const;
    //: Returns true if the range is valid
    //: Empty ranges (ie Min() = Max()+1) are valid
    
    bool Contains(IndexC i) const;
    //: Returns true if this range contains the index 'i'.
    
    bool Contains(RealT val) const;
    //: Is a real value inside the range ?
    
    bool Contains(const IndexRangeC & range) const;
    //: Returns true if this range contains the subrange 'range'.
    
    bool operator==(const IndexRangeC & range) const;
    //: Returns true if both index ranges have the same limits.
    
    bool operator!=(const IndexRangeC & range) const;
    //: Returns true if both the ranges have different limits.
    
    bool In(const IndexRangeC & range) const;
    //: Returns true if this range is inside of the 'range'.
    
     bool IsOverlapping(const IndexRangeC & r) const;
    //: Returns true if this range contains at least one common index with 
    //: the range 'r'.
    
    //:-------------------
    //: Special operations.

    const IndexRangeC &SetOrigin(IndexC position);
    //: Set the origin of the range to 'position'.
    // Returns a reference to this range.
    
    IndexRangeC &operator++();
    //: Move both the max and min of the range along 1.
    // Returns a reference to this range.
    
    IndexRangeC &operator--();
    //: Move both the max and min of the range back 1.
    // Returns a reference to this range.
    
    const IndexRangeC & operator+=(IndexC i);
    //: Both minimum and maximum limits are shifted by adding the offset 'i'.
    // Returns a reference to this range.
    
    const IndexRangeC & operator-=(IndexC i);
    //: Both minimum and maximum limits are shifted by subtracting the offset 'i'.
    // Returns a reference to this range.

    const IndexRangeC & operator /= (IndexC i);
    //: Both minimum and maximum limits are divided by i 
    // IndexC rounding rules apply 
    // Returns a reference to this range. 

    const IndexRangeC & operator *= (IndexC i);
    //: Both minimum and maximum limits are multiplied by i
    // returns a reference to this range 

    const IndexRangeC & operator+=(IntT i);
    //: Both minimum and maximum limits are shifted by adding the offset 'i'.
    // Returns a reference to this range.
    
     const IndexRangeC & operator-=(IntT i)
    { return (*this) -= IndexC(i); }
    //: Both minimum and maximum limits are shifted by subtracting the offset 'i'.
    // Returns a reference to this range.

    const IndexRangeC & operator+=(UIntT i);
    //: Both minimum and maximum limits are shifted by adding the offset 'i'.
    // Returns a reference to this range.
    
    const IndexRangeC & operator-=(UIntT i);
    //: Both minimum and maximum limits are shifted by subtracting the offset 'i'.
    // Returns a reference to this range.
    
    IndexRangeC operator+(IndexC i) const;
    //: Create a new IndexRangeC with minimum and maximum limits shifted by adding the offset 'i'.
    
    IndexRangeC operator-(IndexC i) const;
    //: Create a new IndexRangeC with minimum and maximum limits shifted by subtracting the offset 'i'.
    
    IndexRangeC operator/ (IndexC i) const;
    //: Create a new IndexRangeC with maximum and minimum limits divided by IndexC 
    // IndexC rounding rules apply. 

    IndexRangeC operator* (IndexC i) const;
    //: Create a new IndexRangeC with maximum and minimum limits multiplied by IndexC 

    IndexRangeC operator+(IntT i) const;
    //: Create a new IndexRangeC with minimum and maximum limits shifted by adding the offset 'i'.
    
    IndexRangeC operator-(IntT i) const;
    //: Create a new IndexRangeC with minimum and maximum limits shifted by subtracting the offset 'i'. 

    IndexRangeC operator+(UIntT i) const;
    //: Create a new IndexRangeC with minimum and maximum limits shifted by adding the offset 'i'.
    
    IndexRangeC operator-(UIntT i) const;
    //: Create a new IndexRangeC with minimum and maximum limits shifted by subtracting the offset 'i'.
    
#if RAVL_CPUTYPE_64
     IndexRangeC  operator+(UInt64T i) const;
    //: Create a new IndexRangeC with minimum and maximum limits shifted by adding the offset 'i'.

     IndexRangeC  operator-(UInt64T i) const;
    //: Create a new IndexRangeC with minimum and maximum limits shifted by subtracting the offset 'i'.

     const IndexRangeC & operator-=(UInt64T i);
    //: Both minimum and maximum limits are shifted by subtracting the offset 'i'.
    // Returns a reference to this range.

     const IndexRangeC & operator-=(Int64T i);
    //: Both minimum and maximum limits are shifted by subtracting the offset 'i'.
    // Returns a reference to this range.

     const IndexRangeC & operator+=(UInt64T i);
    //: Both minimum and maximum limits are shifted by adding the offset 'i'.
    // Returns a reference to this range.

     const IndexRangeC & operator+=(Int64T i);
    //: Both minimum and maximum limits are shifted by adding the offset 'i'.
    // Returns a reference to this range.

 #endif

     IndexRangeC operator+(SizeC i) const;
    //: Create a new IndexRangeC with minimum and maximum limits shifted by adding the offset 'i'.

     IndexRangeC operator-(SizeC i) const;
    //: Create a new IndexRangeC with minimum and maximum limits shifted by subtracting the offset 'i'.

     const IndexRangeC & operator+=(SizeC i);
    //: Both minimum and maximum limits are shifted by adding the offset 'i'.
    // Returns a reference to this range.

     const IndexRangeC & operator-=(SizeC i);
    //: Both minimum and maximum limits are shifted by adding the offset 'i'.
    // Returns a reference to this range.

     IndexRangeC & ClipBy(const IndexRangeC & r);
    //: This index range is clipped to contain at most the index range 'r'.

     IndexRangeC & Clip(const IndexRangeC & r) const;
    //: Clip 'r' by this index range and return it.
    
     IndexC Clip(const IndexC & i) const;
    //: The value 'i' is clipped to be within this range.
    
     IndexRangeC FirstHalf() const
    { return IndexRangeC(Min(),Center()); }
    //: Returns the index range < Min(), (Max()+Min()+1)/2 >.
    
     IndexRangeC FirstHalfD() const;
    //: Returns the index range < Min(), (Max()+Min())/2 >.
    
     IndexRangeC Enlarge(IndexC f) const;
    //: Returns the index range whose number of elements is enlarged by
    //: the factor 'f'. The upper limits is changed.
    
     IndexRangeC Expand(IndexC n) const;
    //: Returns the range extended by adding 'n' items on both limits of
    //: this range. 
    
     IndexRangeC Shrink(IndexC n) const;
    //: Returns the range extended by adding 'n' items on both limits of
    //: this range. 
    
     IndexRangeC & ShrinkHigh(IndexC n);
    //: Returns the range shrinked by removing of the 
    //: last 'n' items on both limits of this range. 
    
     IndexRangeC & Swap(IndexRangeC & r);
    //: Exchanges the contents of this range and range 'r'. The function
    //: returns this range. 

    const IndexRangeC &Involve(IndexC i);
    //: Modify this range to ensure index i is contained within it.

    const IndexRangeC &Involve(const IndexRangeC &subRange);
    //: Modify this range to ensure subRange is contained within it.
    
    const IndexRangeC operator+=(const IndexRangeC &subRange);
    //: Add subRange to this one.
    // minI += subRange.minI; minI += subRange.minI <br>
    // returns a reference to this range.
    
    const IndexRangeC operator-=(const IndexRangeC &subRange);
    //: Subtract subRange from this one.
    // minI -= subRange.minI;  minI -= subRange.minI <br>
    // returns a reference to this range.
    
    IndexRangeC operator+(const IndexRangeC &subRange) const;
    //: Add ranges together.
    // returns minI + subRange.minI, maxI + subRange.maxI
    
    IndexRangeC operator-(const IndexRangeC &subRange) const;
    //: Subtract ranges.
    // returns minI - subRange.minI, maxI - subRange.maxI
    
    IndexRangeC AlignWithin(IntT alignment) const;
    //: Return a range within this range that has start and end points which are integer multples of 'alignment' 
    
    SizeT Hash() const;
    //: Generate a hash value for the range.

	%extend 
	{
	 const char *__str__()
      {
        RavlN::StrOStreamC os;
        os << "[" << self->Min() << ", " << self->Max() << "]";
        return os.String().data();
      }
      
      bool __nonzero__()
      {
      	return self->IsValid();
      }
      
      int __len__()
      {
      	return self->Size();
      }
      
	}

  };
  
}

