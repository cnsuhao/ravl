// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html


%include "Ravl/Swig2/DataSetBase.i"

%{
#ifdef SWIGPERL
#undef Copy
#endif

#include "Ravl/PatternRec/DataSet1.hh"
#include "Ravl/PatternRec/SampleVector.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN {
 
  template<typename SampleT> 
  class DataSet1C : public DataSetBaseC {
  public:
  
    typedef typename SampleT::ElementT Element1T;
  
    DataSet1C();
    
    DataSet1C(UIntT sizeEstimate);
    //: Constructor.
    // Constructs an empty dataset, with enough space to hold 'sizeEstimate' elements without and extra allocations.
    
    DataSet1C(const SampleT & dat);
    //: Create a dataset from a sample
    
    SampleT &Sample1();
    //: Access complete sample.
    
    const SampleT &Sample1() const;
    //: Access complete sample.
    
    IndexC Append(const Element1T &data);
    //: Append a data entry.
    // returns its index.

    void Append(const DataSet1C<SampleT> &data);
    //: Append a data set to this one.
    // Note that the elements are not copied
    
    DataSet1C<SampleT> ExtractSample(RealT proportion);
    //: Extract a sample.
    // The elements are removed from this set. NB. The order
    // of this dataset is NOT preserved.

    UIntT Size() const;
    //: Get the size of the dataset.
  
	
  };
  
  %template(DataSet1SampleVectorC) RavlN::DataSet1C<RavlN::SampleVectorC>;
}


