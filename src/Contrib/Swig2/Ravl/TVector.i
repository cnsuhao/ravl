// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html


%include "Ravl/Swig2/SArray1d.i"

%{
#ifdef SWIGPERL
#undef Copy
#endif

#include "Ravl/TVector.hh"
#include "Ravl/SArray1dIter.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN {
	
  template<typename DataT>
  class TVectorC : public SArray1dC<DataT> {
  public:
   
   DataT Product() const;      
    //: Returns the product of all elements of the vector.
    
    DataT SumOfSqr() const;
    //: Returns the sum of the squares of all the elements of the vector.
    
    DataT SumOfAbs() const;
    //: Returns the sum of the absolute values of all the elements of the vector.
    
    DataT MaxValue() const;
    //: Largest value in the array.
    
    DataT MaxMagnitude() const;
    //: Value of the largest magnitude in the vector.
    
    DataT MaxAbsValue() const;
    //: Value of the largest magnitude in the vector.
    //: This is an alias for MaxMagnitude.
    
    DataT MinValue() const;
    //: Smallest value in the array.
    
    DataT MinAbsValue() const;
    //: Smallest absolute value in the array.
    
    const TVectorC<DataT> & Reciprocal();
    //: All elements of the vector are changed to their reciprocal values. 
    //: It is assumed that all elements of the vector differs from zero.
    
    DataT Modulus() const;
    //: Returns the modulus of the vector.
    // The Sqrt(SumOfSqr()).
    
    DataT TMul(const TVectorC<DataT> & b) const;
    //: multiplication 'DataT' = (*this).T() * b
    
    DataT Dot(const TVectorC<DataT> & v) const;         
    //: scalar product of vectors    
    
    DataT Dot2(const TVectorC<DataT> & v1, const TVectorC<DataT> &v2) const;
    //: returns this * v1 + this^2 * v2

    void ElemMul(const TVectorC<DataT> &v2,TVectorC<DataT> &result) const;
    //: Return the element wise product of this vector times v2.

    void ElemSum(const TVectorC<DataT> &v2,TVectorC<DataT> &result) const;
    //: Return the element wise sum of v2 and this vector.

    void ElemSubtract(const TVectorC<DataT> &v2,TVectorC<DataT> &result) const;
    //: Return the element wise of v2 subtracted from this vector.

    const TVectorC<DataT> &SetSmallToBeZero(const DataT &min);
    //: Set values smaller than 'min' to zero in vector.
    
    //TMatrixC<DataT> OuterProduct(const TVectorC<DataT> &a) const;
    //: Calculate the outer product of this vector and a.
    // To use the function you must also include 'Ravl/Matrix.hh'.
    
    //TMatrixC<DataT> OuterProduct(const TVectorC<DataT> &a,DataT b) const;
    //: Calculate the outer product of this vector and a multiplied by b.
    // To use the function you must also include 'Ravl/Matrix.hh'.
    
    //TMatrixC<DataT> OuterProduct() const;
    //: Calculate the outer product of this vector with itself.
    // To use the function you must also include 'Ravl/Matrix.hh'.
    
    TVectorC<DataT> Unit() const;
    //: Return a unit vector
   
    const TVectorC<DataT> &MakeUnit();
    //: Make this a unit vector.
    
    const TVectorC<DataT> &MulAdd(const TVectorC<DataT> & i,DataT a);
    //: Multiply i by a and add it to this vector.
    // Returns a reference to this vector.
    
    //:-
    // Distance calculations
    // ---------------------
    
    DataT MaxValueDistance(const TVectorC<DataT> & i) const;
    //: Returns the distance of two indexes in maximum value metric.
    
    DataT CityBlockDistance(const TVectorC<DataT> & i) const;
    //: Returns the distance of two indexes in absolute value metric.
    
    DataT SqrEuclidDistance(const TVectorC<DataT> & i) const;
    //: Returns the distance of two indexes in square Euclid metric.
    
    DataT EuclidDistance(const TVectorC<DataT> & i) const;
    //: Returns the distance of two indexes in square Euclid metric.
    
    IndexC MaxIndex() const;
    //: Find the index with the most positive valued index.
    
    IndexC MaxAbsIndex() const;
    //: Find the index with the absolute maximum valued index.

    IndexC MinIndex() const;
    //: Find the index with the most negative valued index.
    
    IndexC MinAbsIndex() const;
    //: Find the index with the absolute minimum valued index.
    
    %extend {
	  // Make a vector slightly prettier?    
      const char *__str__()
      {
        RavlN::StrOStreamC os;
        os << std::fixed;
 #if 0
        
        os << "[";
        for(RavlN::SArray1dIterC<DataT>it(self->SArray1d());it;it++) {
        	os << *it;
        	if(!it.IsLast()) 
        		os << ",";
        }
        os << "]";
#else
		os << *self;
#endif
        return PyString_AsString(PyString_FromStringAndSize(os.String().chars(), os.String().Size()));
      }	
    
    }
	

  };
  
  
%template(TVectorReal) RavlN::TVectorC<RavlN::RealT>;
%template(TVectorFloat) RavlN::TVectorC<RavlN::FloatT>;
  
}
