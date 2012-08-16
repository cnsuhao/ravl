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


#include "Ravl/SArray1d.hh"
#include "Ravl/StrStream.hh"
#include "Ravl/Vector.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN {

  template<typename DataT>
  class SArray1dC {
  public:
  
  SArray1dC();
  //: Create an array of zero length

  SArray1dC(const SizeT dim);
  //: Creates an uninitialised array with the range <code>{0 ... dim-1}</code>.

  static SArray1dC<DataT> ConstructAligned(const SizeT dim,UIntT align);
  //: Creates an uninitialised array with the range <code>{0 ... dim-1}</code> and byte alignment of the first element 'align'
  // align must be a power of 2.

  //SArray1dC(const Slice1dC<DataT> &slice,bool alwaysCopy = true);
    //: Make an array from a slice.
    // This will create an array with the values from the slice.
    // if the stride of the slice is not 1, and alwaysCopy is true the
    // a copy is done.

    //SArray1dC(const PairC<DataT> & pr);
    //: Creates an array with two elements from a PairC object.

    SArray1dC(const SArray1dC<DataT> & vv);
    //: Copy constructor.
    // Another access to the array 'vv'.

    SArray1dC(const SArray1dC<DataT> & vv,SizeT dim,SizeT offsetInBuff = 0);
    //: The subarray of the 'vv' with size 'dim'.

    SArray1dC(DataT *data,const SizeT  dim,bool removable);
    //: The array is created from the memory location <code>data</code> with the range
    //: <code>{0 ... dim-1}</code>.
    // <font color="red">Warning:</font>  the <code>data</code> argument is a pointer, with all the attendant problems.
    // The data is <i>not</i> copied.
    //!param: data  - address of the data to be used as the array contents
    //!param: removable - if true, <code>data</code> is  de-allocated from the heap during destruction of the array.<br>If <code>data</code> is not allocated on the heap, this arg <b>MUST</b> set false.<br>N.B>: this arg used to have default value = true

    // <p> It can be used to create an <code>SArray1dC</code> initialised from some constant array, like this:<pre>
    //   static RealT values[9] = {
    //        -0.02311234,   0.00958230,   0.10377361,
    //         0.22375219,   0.27955917,   0.22375219,
    //         0.10377361,   0.00958230,  -0.02311234
    //    };
    //    SArray1dC&lt;RealT&gt; coeffs(values, 9, false);
    //</pre>
    // Here, <code>removable</code> is set <b>false</b> as the data was not allocated on the heap in the first place.<br>
    // Note: it is the programmer's responsibility to make the <code>range</code> argument match the data size.</p>
  
    bool IsEmpty() const;
    // Returns TRUE if the size of the array is zero.
    
    bool Contains(IndexC i) const;
    // Returns TRUE if the array contains an item with the index 'i'.
    
    void Fill(const DataT & d);
    // 'd' value is assigned to all elements of the buffer.
    
    void Reverse();
    //: Reverse the order of elements in this array in place.
    
    SArray1dC<DataT> operator+(const SArray1dC<DataT> & arr) const;
    //: Sums 2 numerical arrays.
    // The operator returns the result as a new array.

    SArray1dC<DataT> operator-(const SArray1dC<DataT> & arr) const;
    //: Subtract 2 numerical arrays.
    // The operator returns the result as a new array.

    SArray1dC<DataT> operator*(const SArray1dC<DataT> & arr) const;
    //: Multiplies 2 numerical arrays.
    // The operator returns the resulhttp://www.swig.org/Doc1.3/Python.html#Python_nn59t as a new array.

    SArray1dC<DataT> operator/(const SArray1dC<DataT> & arr) const;
    //: Divides 2 numerical arrays.
    // The operator returns the result as a new array.

    SArray1dC<DataT> operator*(const DataT &number) const;
    //: Multiplies the array by the 'number'.
    // The operator returns the result as a new array.

    SArray1dC<DataT> operator/(const DataT &number) const;
    //: Divides all array items by the 'number'.
    // The operator returns the result as a new array.

    SArray1dC<DataT> operator+(const DataT &number) const;
    //: Adds 'number' to the array.
    // The operator returns the result as a new array.

    SArray1dC<DataT> operator-(const DataT &number) const;
    //: Subtracts 'number' from all array items.
    // The operator  returns the result as a new array.

    const SArray1dC<DataT> & operator+=(const SArray1dC<DataT> & arr);
    //: Adds the 2nd array to this array.

    const SArray1dC<DataT> & operator-=(const SArray1dC<DataT> & arr);
    //: Subtracts the 2nd array from this array.

    const SArray1dC<DataT> & operator*=(const SArray1dC<DataT> & arr);
    //: Multiplies the 2nd array to this array.

    const SArray1dC<DataT> & operator/=(const SArray1dC<DataT> & arr);
    //: Divides the 2nd array from this array.

    const SArray1dC<DataT> & operator+=(const DataT &number);
    //: Adds 'number' to all array items.

    const SArray1dC<DataT> & operator-=(const DataT &number);
    //: Subtracts 'number' from all array items.

    const SArray1dC<DataT> & operator*=(const DataT &number);
    //: Multiplies the array by the 'number'.

    const SArray1dC<DataT> & operator/=(const DataT &number);
    //: Divides the array elements by the 'number'.

    DataT Sum() const;
    //: Calculate the sum of all elements in the array

    DataT SumOfSqr() const;
    //: Calculate the sum of the squares of all elements in the array

    //IndexC IndexOfMax() const;
    //: Find the index of the maximum element in the array

    //IndexC IndexOfMin() const;
    //: Find the index of the minimum element in the array
    
    const SArray1dC<DataT> & SArray1d() const;
    //: Access to the whole constant array.

    SArray1dC<DataT> & SArray1d();
    //: Access to the whole array.

    SizeT Size() const;
    //: Access size of array

    IndexRangeC Range() const;
    //: Returns the usable range of indices expressed by this object.
    
    SArray1dC<DataT> Join(const SArray1dC<DataT> &Oth) const;
    // Join this Array and another into a new Array which
    // is returned. This does not change either of its arguments.
    // This is placed in the array first, followed by 'Oth'.

    SArray1dC<DataT> Join(const DataT &Oth) const;
    // Join this Array and an element into a new Array which
    // is returned. This does not change either of its arguments.
    // This is placed in the array first, followed by 'Oth'.

    SArray1dC<DataT> & Append(const SArray1dC<DataT> & a);
    // This array is extended by the length of the array 'a' and the contents
    // of both arrays are copied to it. Empty arrays are handled correctly.

    SArray1dC<DataT> & Append(const DataT & a);
    // This array is extended by 1 and the contents of this array are
    // copied to this array followed by the new element.
    // Empty arrays are handled correctly.

    SArray1dC<DataT> From(UIntT offset);
    //: Return array from offset to the end of the array.
    // If offset is larger than the array an empty array
    // is returned,

    SArray1dC<DataT> From(UIntT offset,UIntT size);
    //: Return array from offset to the end of the array.
    // If offset is larger than the array an empty array
    // is returned,

    SArray1dC<DataT> After(UIntT offset);
    //: Return array after offset to the end of the array.
    // If offset is larger than the array an empty array
    // is returned,

    SArray1dC<DataT> Before(UIntT offset)
    { return From(0,offset); }
    //: Return array from the start to the element before offset
    // If offset is larger then the whole array will be returned.
    
    //void QuickSort(typename SArray1dC<DataT>::QuickSortCmpT cmp = &SArray1dC<DataT>::DefaultComparisonOp); 
    //: Sort the array with comparison function 'cmp'.
    // The default is to use the "<" operator; this creates an array sorted in
    // <i>ascending</i> order.<br>
    // Where a comparison operator for DataT does not exist, you must provide
    // your own in place of the default argument.
    // See <a href="../../Examples/exDList.cc.html">example</a> for how to write your own.
    
     bool operator==(const SArray1dC<DataT> & vv) const;
    //: Comparison operator
    // Returns true if the two arrays are the same length and
    // their contents are identical.

    bool operator!=(const SArray1dC<DataT> & vv) const;
    //: Comparison operator
    // Returns true if the two arrays have different lengths or
    // contents..

    SizeT Hash() const;
    //: Compute a hash value for the array.

    DataT * DataStart() const;
    //: Returns the address of element 0.
    // If the array has zero length a null pointer may
    // be returned.
    
	%extend 
	{
	  inline const DataT & __getitem__(size_t i) const { return (*self)[i]; }
	  
	  inline void __setitem__(size_t i, const  DataT & v) { (*self)[i] = v; }
	  
      const char *__str__()
      {
        RavlN::StrOStreamC os;
        os << *self;
        return PyString_AsString(PyString_FromStringAndSize(os.String().chars(), os.String().Size())); 
      }
    }

  };
}



%template(SArray1dUInt) RavlN::SArray1dC<RavlN::UIntT>;
%template(SArray1dReal) RavlN::SArray1dC<RavlN::RealT>;
%template(SArray1dFloat) RavlN::SArray1dC<RavlN::FloatT>;
%template(SArray1dIndexC) RavlN::SArray1dC<RavlN::IndexC>;
%template(SArray1dPoint2dC) RavlN::SArray1dC<RavlN::Point2dC>;
%template(SArray1dVectorC) RavlN::SArray1dC<RavlN::VectorC>;
