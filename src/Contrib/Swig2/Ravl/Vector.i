// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html


%include "Ravl/Swig2/TVector.i"
    
%{
#ifdef SWIGPERL
#undef Copy
#endif

#include "Ravl/Vector.hh"
#include "Ravl/Matrix.hh"
#include "Ravl/SArray1dIter.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN {

  
  class VectorC : public TVectorC<RealT> {
  public:
   
   VectorC();
   // Empty vector
   
    VectorC(SizeT size);
    // Construct from size
    
    VectorC(const SArray1dC<RealT> &oth);
    //: Base class constructor.
    
    VectorC(const TVectorC<RealT> &oth);
    //: Base class constructor.
    
    VectorC(const SArray1dC<FloatT> &oth);
    //: Convert from a float vector.
    
    static VectorC ConstructAligned(const SizeT dim,UIntT align);
    //: Creates an uninitialised array with the range <0, 'dim'-1> and byte alignment of the first element 'align'
    // align must be a power of 2.
    
    VectorC(RealT v1,RealT v2);
    //: Create a vector with two real values.

    VectorC(RealT v1,RealT v2,RealT v3);
    //: Create a vector with three real values.

    VectorC(RealT v1,RealT v2,RealT v3,RealT v4);
    //: Create a vector with four real values.
    
    VectorC(RealT v1,RealT v2,RealT v3,RealT v4,RealT v5);
    //: Create a vector with five real values.

    VectorC(RealT v1,RealT v2,RealT v3,RealT v4,RealT v5,RealT v6);
    //: Create a vector with six real values.
   
   /*
	* Finally the methods from VectorC
	*/
	bool IsReal() const;
    //: Test if vector only contains real values.
    // This will return false if either nan's (Not an number) 
    // or infinite values are found.
    
    /*
    * Functions from SArray1dC involving Vectors.
    * We redefine them to be nice in Python so they remain as Vectors
    * and not the base class.
    */
    
    VectorC operator+(const VectorC & arr) const;
    //: Sums 2 numerical arrays.
    // The operator returns the result as a new array.

    VectorC operator-(const VectorC & arr) const;
    //: Subtract 2 numerical arrays.
    // The operator returns the result as a new array.

    VectorC operator*(const VectorC & arr) const;
    //: Multiplies 2 numerical arrays.
    // The operator returns the result as a new array.

    VectorC operator/(const VectorC & arr) const;
    //: Divides 2 numerical arrays.
    // The operator returns the result as a new array.

    VectorC operator*(const RealT &number) const;
    //: Multiplies the array by the 'number'.
    // The operator returns the result as a new array.

    VectorC operator/(const RealT &number) const;
    //: Divides all array items by the 'number'.
    // The operator returns the result as a new array.

    VectorC operator+(const RealT &number) const;
    //: Adds 'number' to the array.
    // The operator returns the result as a new array.

    VectorC operator-(const RealT &number) const;
    //: Subtracts 'number' from all array items.
    // The operator  returns the result as a new array.

    const VectorC & operator+=(const VectorC & arr);
    //: Adds the 2nd array to this array.

    const VectorC & operator-=(const VectorC & arr);
    //: Subtracts the 2nd array from this array.

    const VectorC & operator*=(const VectorC & arr);
    //: Multiplies the 2nd array to this array.

    const VectorC & operator/=(const VectorC & arr);
    //: Divides the 2nd array from this array.

    const VectorC & operator+=(const RealT &number);
    //: Adds 'number' to all array items.

    const VectorC & operator-=(const RealT &number);
    //: Subtracts 'number' from all array items.

    const VectorC & operator*=(const RealT &number);
    //: Multiplies the array by the 'number'.

    const VectorC & operator/=(const RealT &number);
    //: Divides the array elements by the 'number'.
    
    VectorC Join(const VectorC &Oth) const;
    // Join this Array and another into a new Array which
    // is returned. This does not change either of its arguments.
    // This is placed in the array first, followed by 'Oth'.

    VectorC Join(const RealT &Oth) const;
    // Join this Array and an element into a new Array which
    // is returned. This does not change either of its arguments.
    // This is placed in the array first, followed by 'Oth'.

    VectorC & Append(const VectorC & a);
    // This array is extended by the length of the array 'a' and the contents
    // of both arrays are copied to it. Empty arrays are handled correctly.

    VectorC & Append(const RealT & a);
    // This array is extended by 1 and the contents of this array are
    // copied to this array followed by the new element.
    // Empty arrays are handled correctly.

    VectorC From(UIntT offset);
    //: Return array from offset to the end of the array.
    // If offset is larger than the array an empty array
    // is returned,

    VectorC From(UIntT offset,UIntT size);
    //: Return array from offset to the end of the array.
    // If offset is larger than the array an empty array
    // is returned,

    VectorC After(UIntT offset);
    //: Return array after offset to the end of the array.
    // If offset is larger than the array an empty array
    // is returned,

    VectorC Before(UIntT offset)
    { return From(0,offset); }
    //: Return array from the start to the element before offset
    // If offset is larger then the whole array will be returned.
    
    //void QuickSort(typename VectorC::QuickSortCmpT cmp = &VectorC::DefaultComparisonOp); 
    //: Sort the array with comparison function 'cmp'.
    // The default is to use the "<" operator; this creates an array sorted in
    // <i>ascending</i> order.<br>
    // Where a comparison operator for RealT does not exist, you must provide
    // your own in place of the default argument.
    // See <a href="../../Examples/exDList.cc.html">example</a> for how to write your own.
    
    bool operator==(const VectorC & vv) const;
    //: Comparison operator
    // Returns true if the two arrays are the same length and
    // their contents are identical.

    bool operator!=(const VectorC & vv) const;
    //: Comparison operator
    // Returns true if the two arrays have different lengths or
    // contents..
    
	       
	/*
    * Functions from TVectorC involving Vectors.
    * We redefine them to be nice in Python so they remain as Vectors
    * and not the base class.
    */
    
    const VectorC & Reciprocal();
    //: All elements of the vector are changed to their reciprocal values. 
    //: It is assumed that all elements of the vector differs from zero.
    
	RealT TMul(const VectorC & b) const;
    //: multiplication 'RealT' = (*this).T() * b
    
    RealT Dot(const VectorC & v) const;         
    //: scalar product of vectors    
    
    RealT Dot2(const VectorC & v1, const VectorC &v2) const;
    //: returns this * v1 + this^2 * v2

    void ElemMul(const VectorC &v2,VectorC &result) const;
    //: Return the element wise product of this vector times v2.

    void ElemSum(const VectorC &v2,VectorC &result) const;
    //: Return the element wise sum of v2 and this vector.

    void ElemSubtract(const VectorC &v2,VectorC &result) const;
    //: Return the element wise of v2 subtracted from this vector.

    const VectorC &SetSmallToBeZero(const RealT &min);
    //: Set values smaller than 'min' to zero in vector.
    
    RavlN::MatrixC OuterProduct(const VectorC &a) const;
    //: Calculate the outer product of this vector and a.
    // To use the function you must also include 'Ravl/Matrix.hh'.
    
    RavlN::MatrixC OuterProduct(const VectorC &a,RealT b) const;
    //: Calculate the outer product of this vector and a multiplied by b.
    // To use the function you must also include 'Ravl/Matrix.hh'.
    
    RavlN::MatrixC OuterProduct() const;
    //: Calculate the outer product of this vector with itself.
    // To use the function you must also include 'Ravl/Matrix.hh'.
    
    VectorC Unit() const;
    //: Return a unit vector
   
    const VectorC &MakeUnit();
    //: Make this a unit vector.
    
    const VectorC &MulAdd(const VectorC & i,RealT a);
    //: Multiply i by a and add it to this vector.
    // Returns a reference to this vector.
    
    //:-
    // Distance calculations
    // ---------------------
    
    RealT MaxValueDistance(const VectorC & i) const;
    //: Returns the distance of two indexes in maximum value metric.
    
    RealT CityBlockDistance(const VectorC & i) const;
    //: Returns the distance of two indexes in absolute value metric.
    
    RealT SqrEuclidDistance(const VectorC & i) const;
    //: Returns the distance of two indexes in square Euclid metric.
    
    RealT EuclidDistance(const VectorC & i) const;
    //: Returns the distance of two indexes in square Euclid metric.
    
    %extend
    {
     
    
      PyObject * AsNumPy()
      {
        int nd = 1;
        npy_intp dims[1];
        dims[0]=self->Size().V();
        PyObject * array = PyArray_SimpleNew(nd, dims, PyArray_DOUBLE);
	    RavlN::RealT * dptr = (RavlN::RealT*)PyArray_DATA(array);
        memcpy((void *)dptr, (void *)&(*self)[0], sizeof(RavlN::RealT) * self->Size().V()); 
        return array;
      }
    }
    
   
  };
  
  VectorC RandomVector(int n,RealT scale = 1.0);
  //: Create a random vector of values between -scale and scale with the given size.
  
  inline void SetZero(VectorC &vec);
  //: Fill vector with zero's.
  
  VectorC Sigmoid(const VectorC &z);
  //: Compute the element wise sigmoid of z.

  void SigmoidIP(VectorC &z);
  //: Compute sigmoid values and store the results in place.

  VectorC Log(const VectorC &z);
  //: Compute the element wise log of z.

  void LogIP(VectorC &z);
  //: Compute the element wise log of z and store the results in place.

  VectorC Exp(const VectorC &z);
  //: Compute the element wise exponent of z.
  
  
}
