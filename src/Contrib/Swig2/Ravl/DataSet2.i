// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html


%include "Ravl/Swig2/DataSet1.i"

%{
#ifdef SWIGPERL
#undef Copy
#endif

#include "Ravl/PatternRec/DataSet1.hh"
#include "Ravl/PatternRec/SampleVector.hh"
#include "Ravl/PatternRec/SampleLabel.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN {
 
  template<typename Sample1T, typename Sample2T> 
  class DataSet2C : public DataSet1C<Sample1T> {
  public:
  
  	typedef typename Sample1T::ElementT Element1T;
    typedef typename Sample2T::ElementT Element2T;
  
	DataSet2C();
    //: Default constructor.
    // Creates an invalid handle.
    
    DataSet2C(UIntT sizeEstimate);
    //: Constructor.
    // Constructs an empty dataset, with enough space to hold 'sizeEstimate' elements without and extra allocations.
    
    DataSet2C(const Sample1T & dat1,const Sample2T & dat2);
    //: Create a dataset from some samples.
    
    Sample2T &Sample2();
    //: Access complete sample.
    
    const Sample2T &Sample2() const;
    //: Access complete sample.
    
    IndexC Append(const Element1T &data1,const Element2T &data2);
    //: Append a data entry.
    // returns its index.

    void Append(const DataSet2C<Sample1T,Sample2T> &data);
    //: Append a data set to this one
    // Note that the elements are not copied

    void Append(const SampleC<Element1T> & sample1, const Element2T & element2);
    //: Append a sample of inputs and assign the output as the same for all inputs
    
    //void Append(SampleStream2C<Element1T, Element2T> & data);
    //: Append a sample stream

    //void Append(SampleStreamC<Element1T> & sample1, const Element2T & sample2);
    //: Append a sample stream of inputs and assign the output as the same for all inputs
    
    DataSet2C<Sample1T,Sample2T> ExtractSample(RealT proportion);
    //: Extract a sample.
    // The elements are removed from this set. NB. The order
    // of this dataset is NOT preserved.
  
	
  };

  %template(DataSet2SampleVectorSampleLabelC) RavlN::DataSet2C<RavlN::SampleVectorC, RavlN::SampleLabelC>;

}


