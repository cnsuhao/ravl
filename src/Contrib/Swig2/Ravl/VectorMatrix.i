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

#include "Ravl/VectorMatrix.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN {

  
  class VectorMatrixC  {
  public:
   VectorMatrixC();
	//: Empty object
	    
    VectorMatrixC(const UIntT dim);
    //: Constructor of an empty vector and matrix of the dimension 'dim'.
    
    VectorMatrixC(const VectorC & vec, const MatrixC & mat);  
    //: Constructs the object from the vector 'vec' and the matrix 'mat'.
    //: The new object contains handles of 'vec' and 'matrix' (BIG OBJECT).
    
    VectorMatrixC(const VectorMatrixC &vm);
    //: Copy constructor.
    
    const VectorC & Vector() const;
    //: Access to the vector.
    
    const MatrixC & Matrix() const;
    //: Access to the matrix.
    
    VectorC & Vector();
    //: Non-constant access to the vector.
    
    MatrixC & Matrix();
    //: Non-constant access to the matrix.
    
    VectorMatrixC Copy() const;
    //: Returns the physical copy of this vector matrix.
    
    void SetZero();
    //: Zeros the vector and the matrix.
    
    void Sort();
    //: Sort in place according to the vector value. 
    // The first value will be the  biggest one. 
    
    %extend {
     // Make a vector slightly prettier?    
      const char *__str__()
      {
        RavlN::StrOStreamC os;
        os << *self;
        return PyString_AsString(PyString_FromStringAndSize(os.String().chars(), os.String().Size()));
      }	
    }
    
   
  };
  
 
  
}
