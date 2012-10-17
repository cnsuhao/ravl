// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html


%include "Ravl/Swig2/Sample.i"
%include "Ravl/Swig2/Vector.i"

%{
#ifdef SWIGPERL
#undef Copy
#endif

#include "Ravl/PatternRec/SampleVector.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN {

	enum DataSetNormaliseT {
  		DATASET_NORMALISE_NONE=0,
  		DATASET_NORMALISE_MEAN=1,
  		DATASET_NORMALISE_SCALE=2
	};

  class SampleVectorC : public SampleC<VectorC> {
  	public:
  	
	SampleVectorC(SizeT maxSize=10);
    //: Create a sample of data with a maximum size
    
    SampleVectorC(const SArray1dC<VectorC> & dat);
    //: Create a sample of data from an array
    
    SampleVectorC(const SampleC<VectorC> &svec);
    //: Construct from base class.

    //SampleVectorC(const SampleC<TVectorC<float> > &svec, const SArray1dC<FieldInfoC> & fieldInfo = SArray1dC<FieldInfoC>());
    //: Construct from a sample of floats.

    SampleVectorC(const SampleC<VectorC> &svec, const SArray1dC<IndexC> &featureSet);
    //: Construct a sample set with a reduced set of features
    //!param: svec       - a sample of vectors
    //!param: featureSet - the indexes of features to keep

    SampleVectorC(const MeanCovarianceC & meanCovariance);
    //: Construct a dataset using the statistics and number of samples specified.  Note the actual mean and covariance will differ slighty
    //!param meanCovariance The desired mean and covariance to use to generate the dataset
    
    //SampleVectorC(const XMLFactoryContextC & factory);
    //: Construct a dataset from a factory
    //!param mean the mean vector
    //!param covariance the covariance matrix of the data
    //!param samples the number of samples to generate

    UIntT VectorSize() const;
    //: Get the size of vectors in this sample.
    
    VectorC Mean() const;
    //: Find the mean vector of the sample.
    
    VectorC Mean(const SampleC<RealT> &weights) const;
    //: Find the weighted mean vector of the sample.
    
    MeanCovarianceC MeanCovariance(bool sampleStatistics = true) const;
    //: Find the mean and covariance of the sample
    
    MeanCovarianceC MeanCovariance(const SampleC<RealT> &weights,bool sampleStatistics = true) const;
    //: Find the mean and covariance of a weighted sample
    
    MatrixRUTC SumOuterProducts() const;
    //: Compute the sum of the outerproducts.
    
    MatrixC TMul(const SampleC<VectorC> &sam2) const;
    //: Compute the sum of the outerproducts.
    // sam2 must have the same size as this sample vector.
    
    MatrixRUTC SumOuterProducts(const SampleC<RealT> &w) const;
    //: Compute the sum of the outerproducts weighting each with the corresponding value from 'w'.
    
    MatrixC TMul(const SampleC<VectorC> &sam2,const SampleC<RealT> &w) const;
    //: Compute the sum of the outerproducts weighting each with the corresponding value from 'w'.
    // sam2 must have the same size as this sample vector.
  
  	FunctionC Normalise(DataSetNormaliseT normType);
    //: Apply normalisation to data
     
    void Normalise(const FunctionC & func);
    //: Apply function in place.  Typically used for normalising data.

    //const SArray1dC<FieldInfoC> & FieldInfo() const;
    //: Access information about the fields

    //bool SetFieldInfo(const SArray1dC<FieldInfoC> & fieldInfo);
    //: Set the field information
  	
	
  };
}

