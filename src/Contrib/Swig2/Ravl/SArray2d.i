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

#include "Ravl/SArray2d.hh"
#include "Ravl/StrStream.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN {

  template<typename DataT>
  class SArray2dC {
  public:
  
  	SArray2dC();
    //: Default constructor.
    // Creates a zero size array.

    SArray2dC(SizeT dim1,SizeT dim2);
    //: Constructor.
    // Create a dim1 by dim2 array.

    SArray2dC(const Index2dC &size);
    //: Constructor.
    // Create a size[0] by size[1] array.

    SArray2dC(SArray2dC<DataT> &arr,SizeT size1,SizeT size2);
    //: Construct an access to a sub array of this one.

    SArray2dC(SArray2dC<DataT> &arr,const IndexRange2dC &rng);
    //: Create a new access to 'rng' of 'arr'.
    // 'rng' must be within 'arr'. The origin of the new array will be at 'rng.Origin()' of 'arr'.

    SArray2dC(DataT *data,SizeT size1,SizeT size2,bool copyMemory = false,bool freeMemory = false,IntT stride = 0);
    //: Create size1 x size2 array from memory given in 'data'
    // If freeMemory is true it 'data' will be freed with a 'delete []' call when no longer required.

    static SArray2dC<DataT> ConstructAligned(const SizeT dim1,const SizeT dim2,UIntT align);
    //: Creates an uninitialized array with the range <0, 'dim1'-1>,<0, 'dim2'-1> and
    //: the given byte alignment of the start of each row.
    // align must be a power of 2.
    // Currently the align must be an integer muliple of the element size.

    SArray2dC<DataT> Copy() const;
    //: Copy array.

    SizeT Size1() const;
    //: Range of 1st Index

    SizeT Size2() const;
    //: Range of 2nd index is [0..Size2()-1]

 	bool Contains(const Index2dC &i) const; 
    //: Does this buffer contain the index i ?
    // Returns true if yes.
    
    void Fill(const DataT &d);
    //: Fill array with value.
    
    IntT Stride() const;
    //: Get the stride of the 2d array. 
    
    bool IsContinuous() const;
    //: Test if the array is allocated in a continous area of memory.
    // Note: this only checks the first two rows follow each other in
    // memory, this may miss other discontunities.
    
    IndexRange2dC Frame() const;
    //: Return ranges of indexes

    //:------------------
    // Special operations

    SArray1dC<DataT> AsVector(bool alwaysCopy = false);
    //: Access 2d array as 1d vector.
    // This will only copy the data if the data isn't continuous or
    // alwaysCopy is true, this can make it much more effecient than
    // a straigh copy.

    SArray2dC<DataT> operator+(const SArray2dC<DataT> & arr) const;
    // Sums 2 numerical arrays. The operator returns the result as a new array.

    SArray2dC<DataT> operator-(const SArray2dC<DataT> & arr) const;
    // Subtracts 2 numerical arrays. The operator returns
    // the result as a new array.

    SArray2dC<DataT> operator*(const SArray2dC<DataT> & arr) const;
    // Multiplies 2 numerical arrays. The operator returns the result as a new array.

    SArray2dC<DataT> operator/(const SArray2dC<DataT> & arr) const;
    // Divides 2 numerical arrays. The operator returns
    // the result as a new array.

    SArray2dC<DataT> operator*(const DataT &number) const;
    // Multiplies the array by the 'number'. The operator
    // returns the result as a new array.

    SArray2dC<DataT> operator/(const DataT &number) const;
    // Divides all array items by the 'number'. The operator
    // returns the result as a new array.

    SArray2dC<DataT> operator+(const DataT &number) const;
    // Adds 'number' to the array. The operator
    // returns the result as a new array.

    SArray2dC<DataT> operator-(const DataT &number) const;
    // Subtracts 'number' from all array items. The operator
    // returns the result as a new array.

    const SArray2dC<DataT> & operator+=(const SArray2dC<DataT> & arr);
    // Adds the 2nd array to this array.

    const SArray2dC<DataT> & operator-=(const SArray2dC<DataT> & arr);
    // Subtracts the 2nd array from this array.

    const SArray2dC<DataT> & operator*=(const SArray2dC<DataT> & arr);
    // Multiplies the 2nd array to this array.

    const SArray2dC<DataT> & operator/=(const SArray2dC<DataT> & arr);
    // Divides the 2nd array from this array.

    const SArray2dC<DataT> & operator+=(const DataT &number);
    // Adds 'number' to all array items.

    const SArray2dC<DataT> & operator-=(const DataT &number);
    // Subtracts 'number' from all array items.

    const SArray2dC<DataT> & operator*=(const DataT &number);
    // Multiplies the array by the 'number'.

    const SArray2dC<DataT> & operator/=(const DataT &number);
    // Divides the array elements by the 'number'.

    bool operator==(const SArray2dC<DataT> &op) const;
    //: Do arrays have identical ranges and contents ?

    bool operator!=(const SArray2dC<DataT> &op) const;
    //: Do arrays have different ranges and contents ?

    DataT SumOfSqr() const;
    //: Calculate the sum of the squares of all the elements in the array

    DataT Sum() const;
    //: Returns the sum all elements of the array.

    //Slice1dC<DataT> Diagonal();
    //: Take a slice along the diagonal of the array.

    SArray1dC<DataT> SliceRow(IndexC i);
    //: Access row as 1d array.
    // NB. Changes made to the slice will also affect this array!

    //Slice1dC<DataT> SliceColumn(IndexC i);
    //: Access column as 1d slice.
    // NB. Changes made to the slice will also affect this array!

    void SetColumn(IndexC i,const SArray1dC<DataT> &val);
    //: Set the values in the column i to those in 'val'.
    // 'val' must have a size equal to the number of rows.

    void SetRow(IndexC i,const SArray1dC<DataT> &val);
    //: Set the values in the row i to those in 'val'.
    // 'val' must have a size equal to the number of columns

    //void SetColumn(IndexC i,const Slice1dC<DataT> &val);
    //: Set the values in the column i to those in 'val'.
    // 'val' must have a size equal to the number of rows.

    //void SetRow(IndexC i,const Slice1dC<DataT> &val);
    //: Set the values in the row i to those in 'val'.
    // 'val' must have a size equal to the number of columns

    void SetColumn(IndexC i,const DataT &val);
    //: Set the values in the column i to 'val'.

    void SetRow(IndexC i,const DataT &val);
    //: Set the values in the row i to  'val'.

    void SetSubArray(const Index2dC &origin,const SArray2dC<DataT> &vals);
    //: Set sub array of this one.
    // The origin of 'vals' will be places at 'origin' of this array.
    // NOTE: all of vals must fit within this array.

    SizeT Hash() const;
    //: Generate hash for contents of array.

	%extend 
	{
	  //inline const DataT & __getitem__(size_t i) const { return (*self)[i]; }
	  
	  //inline void __setitem__(size_t i, const  DataT & v) { (*self)[i] = v; }
	  
      const char *__str__()
      {
        RavlN::StrOStreamC os;
        os << *self;
        return PyString_AsString(PyString_FromStringAndSize(os.String().chars(), os.String().Size()));    
      }
    }

  };
}

%template(SArray2dFloat) RavlN::SArray2dC<RavlN::FloatT>;
%template(SArray2dReal) RavlN::SArray2dC<RavlN::RealT>;

