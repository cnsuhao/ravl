// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html



%include "Ravl/Swig2/DataSet2.i"
%include "Ravl/Swig2/SampleVector.i"
%include "Ravl/Swig2/SampleLabel.i"

%{
#ifdef SWIGPERL
#undef Copy
#endif

#include "Ravl/PatternRec/DataSetVectorLabel.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}


namespace RavlN {

  class DataSetVectorLabelC : public DataSet2C<SampleVectorC, SampleLabelC> {
public:
	DataSetVectorLabelC();
    //: Default constructor.

    DataSetVectorLabelC(const DataSet2C<SampleVectorC,SampleLabelC> &dataSet);
    //: base class constructor

    DataSetVectorLabelC(UIntT sizeEstimate);
    //: Construct from an estimate of the size.
    
    DataSetVectorLabelC(const SampleVectorC & vec,const SampleLabelC & lab);
    //: Constructor

    DataSetVectorLabelC(const SArray1dC<MeanCovarianceC> & stats);
    //: Constructor

    //DataSetVectorLabelC(const XMLFactoryContextC & factory);
    //: Constructor

    SArray1dC<SampleVectorC> SeperateLabels() const;
    //: Create a separate sample for each label.

    VectorC GlobalMean() const;
    //: returns mean of the input vectors

    SArray1dC<VectorC> ClassMeans () const;
    //: Returns mean of input vectors for each label value

    DataSetVectorLabelC ClassMeansLabels () const;
    //: Returns mean of input vectors for each label along with the label

    SArray1dC<UIntT> ClassNums () const;
    //: Returns array containing the count of each label value

    SArray1dC<MeanCovarianceC> ClassStats (bool sampleStatistics = true) const;
    //: Returns mean and covariance of input vectors for each label value

    MatrixC BetweenClassScatter () const;
    //: Returns between class scatter (covariance) matrix
    // This assumes the prior probabilities of the classes are reflected in 
    // the number of samples in each set.
    
    MatrixC WithinClassScatter (bool sampleStatistics = false) const;
    //: Returns within class scatter (covariance) matrix
    // This assumes the prior probabilities of the classes are reflected in 
    // the number of samples in each set.
    
    DataSetVectorLabelC ExtractPerLabel(UIntT numSamples) const;
    //: Extracts numSamples samples per label
    
    DataSetVectorLabelC ExtractSample(RealT proportion);
    //: Extract a sample.
    // The elements are removed from this set. NB. The order
    // of this dataset is NOT preserved.
    
    SampleVectorC & Sample1();
    //: Access complete sample.
    
    const SampleVectorC &Sample1() const;
    //: Access complete sample.
    
    SampleLabelC & Sample2();
    //: Access complete sample.
    
    const SampleLabelC &Sample2() const;
    //: Access complete sample.
    	
    %extend {
     const char *__str__()
      {
        RavlN::StrOStreamC os;
        os << self->Sample1();
        os << self->Sample2();
        return PyString_AsString(PyString_FromStringAndSize(os.String().chars(), os.String().Size())); 
      }
     }
   
  };
  
  DataSetVectorLabelC CreateDataSet(UIntT dimension = 2 , UIntT classes = 2, UIntT samplesPerClass = 1000, RealT dist = 3.0);
  
}
