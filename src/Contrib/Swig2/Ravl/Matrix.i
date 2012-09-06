// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html


%include "Ravl/Swig2/TMatrix.i"
%include "Ravl/Swig2/Vector.i"

%{
#ifdef SWIGPERL
#undef Copy
#endif

#include "Ravl/Matrix.hh"
#include "Ravl/Vector.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN {


  class MatrixC : public TMatrixC<RealT> {
  public:
    MatrixC();
    //: Default constructor.

    bool IsReal() const;
    //: Test if matrix only contains real values.
    // This will return false if either nan's (Not an number) 
    // or infinite values are found.

    MatrixC(const VectorC &vec);
    //: Treat vector as column matrix.
    
    MatrixC(UIntT rows,UIntT cols);
    //: Construct a new matrix of rows x cols.
    
    MatrixC(UIntT rows,UIntT cols,const RealT *data);
    //: Construct a new matrix of rows x cols with row wise array of data.

    MatrixC(UIntT rows,UIntT cols,const RealT &data);
    //: Construct a new matrix, fill with copy of 'data'.
    
    MatrixC(UIntT rows,UIntT cols,VectorC &data);
    //: Convert an array into a rows by cols matrix.

    MatrixC(SizeT rows,SizeT cols,const VectorC &data);
    //: Convert an array into a rows by cols matrix.

    MatrixC(const TMatrixC<RealT> &oth);
    //: Base class constructor.
    
    MatrixC(const SArray2dC<RealT> &oth);
    //: Base class constructor.
    
    MatrixC(RealT v1,RealT v2,
	    RealT v3,RealT v4);
    //: Construct a 2 x 2 matrix from given values.
    
    MatrixC(RealT v1,RealT v2,RealT v3,
	    RealT v4,RealT v5,RealT v6,
	    RealT v7,RealT v8,RealT v9);
    //: Construct a 3 x 3 matrix from given values.
    
    MatrixC(const SArray2dC<FloatT> &oth);
    //: Convert from a float vector.
    
    
    MatrixC Inverse() const;
    //: Calculate the inverse of this matrix.
    // an invalid matrix (with 0 elements) is returned if this matrix is
    // singular. 
    
    bool InverseIP(RealT &det);
    //: Calculate the inverse of this matrix and its determinant in place.
    // Returns false if matrix is singular. <p>
    // Notes:
    // The matrix must be square <p>
    // If the matrix is not stored in a continuous area of memory a slightly
    // different routine is used to do the inversion.      
    
    bool InverseIP();
    //: Calculate the inverse of this matrix in place.
    // Returns false if matrix is singular. <p>
    // Notes:
    // The matrix must be square <p>
    // If the matrix is not stored in a continuous area of memory a slightly
    // different routine is used to do the inversion.  
    
    MatrixC PseudoInverse(RealT thresh = 1e-5) const;
    //: Calculate the pseudo inverse 
    // <p>Uses <a href="RavlN.SVDC.html">singular value decomposition</a> to
    // decompose the matrix, and sets
    // the singular values smaller than 'thresh' to zero.</p>
    // <p>If the m x n matrix is not square, ensure m &gt; n.</p>
    // <p>If the rank of the matrix M is r, and the pseudo-inverse is denoted
    // M~, then M~ * M is a unit matrix of size r.  On the other hand, M * M~
    // is not a unit matrix.</p>

    RealT Det() const;
    //: Calculate the determinant of the matrix.
    
    void NormaliseRows();
    //: Normalise rows so they have a magnitude of 1.
    // Zero rows are ignored.
    
    void NormaliseColumns();
    //: Normalise columns so they have a magnitude of 1.
    // Zero rows are ignored.
    
    const TMatrixC<RealT> &AddOuterProduct(const TVectorC<float> &vec1,const TVectorC<float> &vec2);
    //: Add outer product of vec1 and vec2 to this matrix.

    const TMatrixC<RealT> &AddOuterProduct(const TVectorC<float> &vec1,const TVectorC<float> &vec2,const float &a);
    //: Add outer product of vec1 and vec2 multiplied by a to this matrix .

	/*
	* From SArray2d 
	*/
	VectorC AsVector(bool alwaysCopy = false);
    //: Access 2d array as 1d vector.
    // This will only copy the data if the data isn't continuous or
    // alwaysCopy is true, this can make it much more effecient than
    // a straigh copy.

    MatrixC operator+(const MatrixC & arr) const;
    // Sums 2 numerical arrays. The operator returns the result as a new array.

    MatrixC operator-(const MatrixC & arr) const;
    // Subtracts 2 numerical arrays. The operator returns
    // the result as a new array.

    MatrixC operator*(const MatrixC & arr) const;
    // Multiplies 2 numerical arrays. The operator returns the result as a new array.

    MatrixC operator/(const MatrixC & arr) const;
    // Divides 2 numerical arrays. The operator returns
    // the result as a new array.

    MatrixC operator*(const RealT &number) const;
    // Multiplies the array by the 'number'. The operator
    // returns the result as a new array.

    MatrixC operator/(const RealT &number) const;
    // Divides all array items by the 'number'. The operator
    // returns the result as a new array.

    MatrixC operator+(const RealT &number) const;
    // Adds 'number' to the array. The operator
    // returns the result as a new array.

    MatrixC operator-(const RealT &number) const;
    // Subtracts 'number' from all array items. The operator
    // returns the result as a new array.

    const MatrixC & operator+=(const MatrixC & arr);
    // Adds the 2nd array to this array.

    const MatrixC & operator-=(const MatrixC & arr);
    // Subtracts the 2nd array from this array.

    const MatrixC & operator*=(const MatrixC & arr);
    // Multiplies the 2nd array to this array.

    const MatrixC & operator/=(const MatrixC & arr);
    // Divides the 2nd array from this array.

    const MatrixC & operator+=(const RealT &number);
    // Adds 'number' to all array items.

    const MatrixC & operator-=(const RealT &number);
    // Subtracts 'number' from all array items.

    const MatrixC & operator*=(const RealT &number);
    // Multiplies the array by the 'number'.

    const MatrixC & operator/=(const RealT &number);
    // Divides the array elements by the 'number'.

    bool operator==(const MatrixC &op) const;
    //: Do arrays have identical ranges and contents ?

    bool operator!=(const MatrixC &op) const;
    //: Do arrays have different ranges and contents ?

    RealT SumOfSqr() const;
    //: Calculate the sum of the squares of all the elements in the array

    RealT Sum() const;
    //: Returns the sum all elements of the array.

    //Slice1dC<RealT> Diagonal();
    //: Take a slice along the diagonal of the array.

    VectorC SliceRow(IndexC i);
    //: Access row as 1d array.
    // NB. Changes made to the slice will also affect this array!

    //Slice1dC<RealT> SliceColumn(IndexC i);
    //: Access column as 1d slice.
    // NB. Changes made to the slice will also affect this array!

    void SetColumn(IndexC i,const VectorC &val);
    //: Set the values in the column i to those in 'val'.
    // 'val' must have a size equal to the number of rows.

    void SetRow(IndexC i,const VectorC &val);
    //: Set the values in the row i to those in 'val'.
    // 'val' must have a size equal to the number of columns

    //void SetColumn(IndexC i,const Slice1dC<RealT> &val);
    //: Set the values in the column i to those in 'val'.
    // 'val' must have a size equal to the number of rows.

    //void SetRow(IndexC i,const Slice1dC<RealT> &val);
    //: Set the values in the row i to those in 'val'.
    // 'val' must have a size equal to the number of columns

    void SetColumn(IndexC i,const RealT &val);
    //: Set the values in the column i to 'val'.

    void SetRow(IndexC i,const RealT &val);
    //: Set the values in the row i to  'val'.

    void SetSubArray(const Index2dC &origin,const MatrixC &vals);
    //: Set sub array of this one.
    // The origin of 'vals' will be places at 'origin' of this array.
    // NOTE: all of vals must fit within this array.

    SizeT Hash() const;
    //: Generate hash for contents of array.
	
	
	/*
	* From TMatrix
	*/

    VectorC operator*(const VectorC & vector) const;
    //: Multiplication "VectorC" = "This" * vector
       
    MatrixC MulT(const MatrixC & B) const;
    //: Multiplication A * B.T()
    
    MatrixC TMul(const MatrixC & B) const;
    //: Multiplication A.T() * B
    // Note: Because of effects of memory layout it this is 
    // approximately half the speed of MulT().
    
    VectorC TMul(const VectorC& vec) const;
    //: Multiplication A.T() * vec
    
    MatrixC AAT() const;
    //: Return  A * A.T().
    
    MatrixC ATA() const;
    //: Return  A.T() * A.
    // Note: Because of effects of memory layout this is much
    // slower than AAT(). It may even be worth using T().AAT()
    // over this call.
    
    MatrixC T() const;
    //: Get transpose of matrix.
    
    static MatrixC Identity(UIntT n);
    //: Returns an identity matrix of n by n.
    // NB. This is a static function and should be called  MatrixC::Identity(n).
    // where n is the size of the matrix.
    
    const MatrixC &SetDiagonal(const VectorC &d);
    //: Set the diagonal of this matrix.
    // If d.Size() != Cols() an error is given.

    const MatrixC &AddDiagonal(const VectorC &d);
    //: Add a vector to the diagonal of this matrix.
    // If d.Size() != Cols() an error is given.
    
    MatrixC SubMatrix(SizeT size1,SizeT size2);
    //: Get sub matrix of size1,size2.
    // The creates a new access, but does not copy the data itself.
    // The matrix always starts from 0,0.
        
    const MatrixC &AddOuterProduct(const VectorC &vec1,const VectorC &vec2);
    //: Add outer product of vec1 and vec2 to this matrix.
    
    const MatrixC &AddOuterProduct(const VectorC &vec1,const VectorC &vec2,const RealT &a);
    //: Add outer product of vec1 and vec2 multiplied by a to this matrix .
    
    const MatrixC &SetSmallToBeZero(const RealT &min);
    //: Set values smaller than 'min' to zero in vector.
    
    const MatrixC &MulAdd(const MatrixC & i,RealT a);
    //: Multiply i by a and add it to this matrix.
    // Returns a reference to this matrix.
    

  };
  

  VectorC Solve(const MatrixC &A, const VectorC &b);
  //: Solve a general linear system  A*<b>x</b> = <b>b</b>
  // The solution vector <b>x</b> is the return value.<br>
  // If A is singular a zero length vector is returned.
  
  bool SolveIP(MatrixC &A, VectorC &b);
  //: Solve a general linear system  A*<b>x</b> = <b>b</b> in place.
  // The vector <b>b</b> is replaced by the solution vector <b>x</b>. <br> 
  // The matrix A is replaced by the L-U factored form. <br> 
  // If A is singular, the return value is false; otherwise the return value is true.<br> 
  
  MatrixC Solve(const MatrixC &A, const MatrixC &B);
  //: Solve a general linear system  A*X = B
  // The solution matrix is the return value.<br>
  // If A is singular a zero-sized matrix is returned.
  
  bool SolveIP(MatrixC &A, MatrixC &B);
  //: Solve a general linear system  A*X = B in place. 
  // The matrix of vectors B is replaced by the solution matrix X. <br> 
  // The matrix A is replaced by the L-U factored form. <br> 
  // If A is singular, the return value is false; otherwise the return value is true.<br> 
  
  VectorC SVD(const MatrixC &M);
  //: Singular value decomposition, eg. M = U * S * V.T(). 
  // The diagonal matrix S is returned as a vector. Values for the
  // other matrices are not computed.
  // If the operation fails the returned vector is invalid.
  
  VectorC SVD_IP(MatrixC &M);
  //: Singular value decomposition, eg. M = U * S * V.T(). 
  // The diagonal matrix S is returned as a vector. Values for the
  // other matrixes are not computed.
  // If the operation fails the returned vector is invalid. <p>
  // NB. This function destroys the contents of this matrix.
  
  VectorC SVD(const MatrixC &M,MatrixC & u, MatrixC & v);
  //: Singular value decomposition, eg. M = U * S * V.T(). 
  // The diagonal matrix S is returned as a vector.<br>
  // This also returns the U and V matrices.  The passed
  // matrices will be used to store the results if they are
  // of the correct size; otherwise new matrices will be allocated.<br>
  // If the operation fails the returned vector is invalid.
  
  VectorC SVD_IP(MatrixC &M,MatrixC & u, MatrixC & v);
  //: Singular value decomposition, eg. M = U * D * V.T(). 
  // The diagonal matrix D is returned as a vector.
  // This also returns the U and V matrices.  The passed
  // matrixes will be used to store the results if they are
  // of the correct size; otherwise new matrices will be allocated.<br>
  // If the operation fails the returned vector is invalid.<br>
  // NB. This function destroys the contents of this matrix.
  
  VectorC EigenValues(const MatrixC &M);
  //: Calculate the eigenvalues of this matrix, for real symmetric matrices
  // This matrix remains unchanged.  <br>
  // A vector of the eigen values is returned.
  // If any errors occurred a zero length vector is generated.
  
  VectorC EigenValuesIP(MatrixC &M);
  //: Calculate the eigenvalues of this matrix, for real symmetric matrices
  // The contents of this matrix is destroyed, but less memory is needed than for <a href="RavlN.EigenValuesObconst_MatrixC_AmpCb.html">EigenValues(const MatrixC &M)</a>.  <br>
  // A vector of the eigen values is returned. 
  // If any errors occurred a zero length vector is generated.
  
  VectorC FastEigenValues(MatrixC &M);
  //: Calculate the eigenvalues of this matrix, for real symmetric matrices
  // As <a href="RavlN.EigenValuesIPObMatrixC_AmpCb.html">EigenValuesIP(MatrixC &M)</a>, but uses a faster algorithm (from CCMath) that occasionally fails for ill-conditioned matrices.<br>
  // The contents of this matrix is destroyed.  <br>
  // A vector of the eigen values is returned.  
  // If any errors occur a zero length vector is generated.<br>
  
  VectorC EigenVectors(const MatrixC &M,MatrixC &E);
  //: Calculate the eigenvalues and vectors of this matrix, for real symmetric matrices.
  // M = E*L*E~ where L is the diagonal matrix of eigenvalues.<br>
  // The matrix M remains unchanged. <br>
  // L is returned as a VectorC.
  
  //VectorMatrixC EigenVectors(const MatrixC &M);
  //: Calculate the eigenvalues and vectors of this matrix, for real symmetric matrices.
  // M = E*L*E~ where L is the diagonal matrix of eigenvalues.<br>
  // The matrix M remains unchanged. <br>
  // L and E are returned as the vector and matrix components respectively of the returned VectorMatrixC.
  
  VectorC EigenVectorsIP(MatrixC &M);
  //: Calculate the eigenvalues and vectors of this matrix, for real symmetric matrices
  // As <a href="RavlN.EigenVectorsObconst_MatrixC_AmpCb.html">EigenVectors(MatrixC &M)</a>, except that eigenvalue matrix E is returned through the argument A.<br>
  // The eigenvalues L are returned as VectorC.

  VectorC FastEigenVectors(MatrixC &M);
  //: Calculate the eigen values and vectors of this matrix, for real symmetric matrices
  // As <a href="RavlN.EigenVectorsIPObMatrixC_AmpCb.html">EigenVectorsIP(MatrixC &M)</a>, but uses a faster algorithm (from CCMath) that occasionally fails for ill-conditioned matrices.<br>
  // The eigenvalue matrix E is returned through the argument M.<br>
  // L is returned as a VectorC.
  // If any errors occur a zero length vector is generated.<br>

  RealT MaxEigenValue(const MatrixC &M,VectorC &maxv);
  //: Get the maximum eigen value and its vector, for real symmetric matrices
  
  //! docentry="Ravl.API.Math.Linear Algebra"
  MatrixC RandomMatrix(int n,int m,RealT scale = 1);
  //: Create a random matrix of values between -scale and scale with the given size.
  
  MatrixC RandomSymmetricMatrix(int n,RealT scale = 1);
  //: Create a random symmetric matrix of values between -scale and scale with the given size.
  
  MatrixC RandomPositiveDefiniteMatrix(int n);
  //: Create a random positive definite matrix.
  // The matrix is also symmetric in the current implementation, this may be changed at some point.
  
}
