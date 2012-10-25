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

#include "Ravl/PatternRec/SampleLabel.hh"
#include "Ravl/RCHash.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN {

  class SampleLabelC : public SampleC<UIntT> {
  	
public:
   	  SampleLabelC(SizeT maxSize = 10);
      //: Create a sample of data with a maximum size

      SampleLabelC(const SArray1dC<UIntT> & dat);
      //: Create a sample of data from an array

      SampleLabelC(const SampleC<UIntT> &sample);
      //: Construct from base class.

      UIntT MaxValue() const;
      //: Find the value of the largest label.

      SArray1dC<UIntT> LabelSums() const;
      //: Return the number of samples in each class

      SampleC<VectorC> SampleVector(RealT inClass = 1, RealT outClass = 0, IntT maxLabel = -1) const;
      //: Convert a sample of labels to vectors
      // Where the label index is set to 'inClass' and the rest to 'outClass'.

      //const FieldInfoC & FieldInfo() const;
      //: Access field info

      //bool SetFieldInfo(const FieldInfoC & fieldInfo);
      //: Set the field info

      bool SetClassName(UIntT label, const StringC & className);
      //: Map a label to a class name

      bool GetClassName(UIntT label, StringC & className);
      //: Get a class name

      const RavlN::RCHashC<UIntT, StringC> & Label2ClassNames() const;
      //: Get the map of label to class names
	
	 %extend
    {
      PyObject * AsNumPy()
      {
        int nd = 1;
        npy_intp dims[1];
        dims[0]=self->Size();
        PyObject * array = PyArray_SimpleNew(nd, dims, PyArray_DOUBLE);
	    RavlN::RealT * dptr = (RavlN::RealT*)PyArray_DATA(array);
		for(RavlN::SampleIterC<RavlN::UIntT>it(*self);it;it++) {
			*dptr=(RavlN::RealT)*it;
			dptr++;
		}
        return array;
      }
    }
	
	
  };
}

