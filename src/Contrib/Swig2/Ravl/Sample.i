// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html


%include "Ravl/Swig2/Vector.i"

%{
#ifdef SWIGPERL
#undef Copy
#endif

#include "Ravl/PatternRec/Sample.hh"
#include "Ravl/Vector.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN {

  template<typename DataT>	
  class SampleC {
  public:
   SampleC(SizeT maxSize=10);
   //: Create a sample of data with a maximum size
    
    SampleC(const SArray1dC<DataT> & dat);
    //: Create a sample of data from an array
    // Note: The data is NOT copied; any operations done
    // on the sample may effect the contents of the original array.
    
    SampleC<DataT> Copy (void);
    //: Make a copy of this object 

    //SampleC<DataT> SubSample(const CollectionC<UIntT> &x);
    //: Take a subsample of the given indexes in x.
    
    SampleC<DataT> CompactFrom(IndexC start,SizeT size);
    //: Get sub array from this one.
    // The new array will be indexed from zero and continous
    // This does not copy the elements, it only creates a new access to existing ones.

    const SampleC<DataT> CompactFrom(IndexC start,SizeT size) const;
    //: Get sub array from this one.
    // The new array will be indexed from zero and continous
    // This does not copy the elements, it only creates a new access to existing ones.

    DataT ExtractEntry(int ind);
    //: Extract an entry from sample.

    IndexC Append(const DataT & dat);
    //: Insert a single sample into sample
    
    void Fill(const DataT &value);
    //: Fill sample array with value.
    
    IndexC Append(const SampleC<DataT> &newData);
    //: Append data to this array.
    // Note: The data is NOT copied; any operations done
    // on the sample may effect the contents of the original array. <br>
    // The number of items appended is returned.
    
    IndexC operator+=(const DataT & dat);
    //: Indentical to Append().
    
    SizeT Size() const;
    //: Return the number of valid samples in the collection
    
    DataT PickElement(UIntT i);
    //: Pick a item i from the collection
    // Note: The order of the collection is NOT preserved.
    // This minimizes the distruption to the underlying
    // representation by removing an element from the end
    // of the array and placing it in the hole left by 
    // removing 'i'.

    DataT &Nth(UIntT i);
    //: Access nth element in sample.
    
    const DataT &Nth(UIntT i) const;
    //: Access nth element in sample.
    
    DataT Pick();
    //: Pick a random item from the collection
    // Note: The order of the collection is NOT preserved.
    
    const DataT & Sample(void) const;
    //: Access a random sample (const)  
    // The sample is NOT removed 

    inline DataT & Sample (void) ;
    //: Access a random sample 
    // The sample is NOT removed 
  
    DataT &First();
    //: Access first element in the array.
    
    const DataT &First() const;
    //: Access first element in the array.
    
    DataT &Last();
    //: Access last element in the array.
    
    const DataT &Last() const;
    //: Access last element in the array.
    
    bool IsEmpty() const;
    //: Is this empty ?
    
    bool IsValid() const;
    //: Is this a valid handle ?
	
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
  
%template(SampleOfUInt) RavlN::SampleC<RavlN::UIntT>;
%template(SampleOfVectorC) RavlN::SampleC<RavlN::VectorC>;
}


