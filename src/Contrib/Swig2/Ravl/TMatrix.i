// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html


%include "Ravl/Swig2/SArray2d.i"

%{
#ifdef SWIGPERL
#undef Copy
#endif

#include "Ravl/TMatrix.hh"
#include "Ravl/TVector.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN {
	
  template<typename DataT>
  class TMatrixC : public SArray2dC<DataT> {
  public:
    TMatrixC();
    //: Default constructor.

    TMatrixC(const SArray2dC<DataT> &oth);
    //: Base class constructor.
    
    TMatrixC(const TVectorC<DataT> &vec);
    //: Treat vector as column matrix.
    // Note: This does not copy the vector, changes
    // made to the matrix will appear in the vector.
    
    TMatrixC(SizeT rows,SizeT cols);
    //: Constructor.
    
    TMatrixC(SizeT rows,SizeT cols,const DataT *data);
    //: Constructor.
    // With row wise array of initialisations data.
    
    TMatrixC(SizeT rows,SizeT cols,DataT *data,bool useCopy,bool manageMemory = false);
    //: Constructor.
    // This allows 'data' to be used in the array.  
    // If 'useCopy' is true the 'manageMemory' flag has no effect.
    
    TMatrixC(SizeT rows,SizeT cols,const DataT &data);
    //: Constructor.
    // Fill the matrix with 'data'..
    
    TMatrixC(SizeT rows,SizeT cols,const SArray1dC<DataT> &data,SizeT stride = 0);
    //: Convert an array into a rows by cols matrix.
    
    TMatrixC(DataT v1,DataT v2,
	     DataT v3,DataT v4);
    //: Construct a 2 x 2 matrix from given values.
    
    TMatrixC(DataT v1,DataT v2,DataT v3,
	     DataT v4,DataT v5,DataT v6,
	     DataT v7,DataT v8,DataT v9);
    //: Construct a 3 x 3 matrix from given values.
    
    
    SizeT Rows() const;
    //: Return the number of rows
    
    SizeT Cols() const;
    //: Return the number of columns
    
    TMatrixC<DataT> operator*(DataT val) const;
    //: Multiply by a constant.

    TVectorC<DataT> operator*(const TVectorC<DataT> & vector) const;
    //: Multiplication "TVectorC<DataT>" = "This" * vector
    
    TMatrixC<DataT> operator*(const TMatrixC<DataT> & mat) const;
    //: Multiplication "result" = "this" * "mat"
    
    TMatrixC<DataT> MulT(const TMatrixC<DataT> & B) const;
    //: Multiplication A * B.T()
    
    TMatrixC<DataT> TMul(const TMatrixC<DataT> & B) const;
    //: Multiplication A.T() * B
    // Note: Because of effects of memory layout it this is 
    // approximately half the speed of MulT().
    
    TVectorC<DataT> TMul(const TVectorC<DataT>& vec) const;
    //: Multiplication A.T() * vec
    
    TMatrixC<DataT> AAT() const;
    //: Return  A * A.T().
    
    TMatrixC<DataT> ATA() const;
    //: Return  A.T() * A.
    // Note: Because of effects of memory layout this is much
    // slower than AAT(). It may even be worth using T().AAT()
    // over this call.
    
    TMatrixC<DataT> T() const;
    //: Get transpose of matrix.
    
    static TMatrixC<DataT> Identity(UIntT n);
    //: Returns an identity matrix of n by n.
    // NB. This is a static function and should be called  MatrixC::Identity(n).
    // where n is the size of the matrix.
    
    const TMatrixC<DataT> &SetDiagonal(const TVectorC<DataT> &d);
    //: Set the diagonal of this matrix.
    // If d.Size() != Cols() an error is given.

    const TMatrixC<DataT> &AddDiagonal(const TVectorC<DataT> &d);
    //: Add a vector to the diagonal of this matrix.
    // If d.Size() != Cols() an error is given.
    
    TMatrixC<DataT> SubMatrix(SizeT size1,SizeT size2)
    { return TMatrixC<DataT>(SArray2dC<DataT>(*this,size1,size2)); }
    //: Get sub matrix of size1,size2.
    // The creates a new access, but does not copy the data itself.
    // The matrix always starts from 0,0.
    
    DataT SumOfAbs() const;
    //: Sum the absolute values of all members of the matrix.
    
    const TMatrixC<DataT> &AddOuterProduct(const TVectorC<DataT> &vec1,const TVectorC<DataT> &vec2);
    //: Add outer product of vec1 and vec2 to this matrix.
    
    const TMatrixC<DataT> &AddOuterProduct(const TVectorC<DataT> &vec1,const TVectorC<DataT> &vec2,const DataT &a);
    //: Add outer product of vec1 and vec2 multiplied by a to this matrix .
    
    const TMatrixC<DataT> &SetSmallToBeZero(const DataT &min);
    //: Set values smaller than 'min' to zero in vector.
    
    const TMatrixC<DataT> &MulAdd(const TMatrixC<DataT> & i,DataT a);
    //: Multiply i by a and add it to this matrix.
    // Returns a reference to this matrix.
    
    void SwapRows(int i,int j);
    //: Swap two rows in the matrix.
   

  };
  
  
%template(TMatrixReal) RavlN::TMatrixC<RavlN::RealT>;
%template(TMatrixFloat) RavlN::TMatrixC<RavlN::FloatT>;
  
}
