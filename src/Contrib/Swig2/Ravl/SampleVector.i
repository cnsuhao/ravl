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
    
    void Normalise(const MeanCovarianceC & stats);
    //: Normalises the input vectors using given stats, in place
    // In order to achieve zero mean and unity variance this function should be
    // called with the return value from MeanCovariance. Subsequent data sets can
    // then be normalised the same way by recording the MeanCovarianceC returned by
    // MeanCovariance.
    
    void Normalise(const FunctionC & func);
    //: Apply function in place.  Typically used for normalising data.

    void UndoNormalisation(const MeanCovarianceC & stats);
    //: Undo the normalisation done by 'Normalise()', in place.

    void Normalise(const MeanCovarianceC & stats,SampleVectorC &sampleVector,bool addBiasElement = false) const;
    //: Normalises the input vectors using given stats, append results to sampleVector
    // In order to achieve zero mean and unity variance this function should be
    // called with the return value from MeanCovariance. Subsequent data sets can
    // then be normalised the same way by recording the MeanCovarianceC returned by
    // MeanCovariance.
    
    void UndoNormalisation(const MeanCovarianceC & stats,SampleVectorC &sampleVector,bool removeBiasElement = false) const;
    //: Undo the normalisation done by 'Normalise()', append results to sampleVector

    //FuncMeanProjectionC NormalisationFunction(const MeanCovarianceC & stats) const;
    //: Get the function used for the normalisation

    //FuncLinearC UndoNormalisationFunction(const MeanCovarianceC & stats) const;
     //: Get the function used to un-normalise the data


    //void Scale(FuncLinearC & func);
    //: Compute function that scales each dimension between 0 and 1 and return function created to do this
    //: !param: func The function that performs the scaling

    //const SArray1dC<FieldInfoC> & FieldInfo() const;
    //: Access information about the fields

    //bool SetFieldInfo(const SArray1dC<FieldInfoC> & fieldInfo);
    //: Set the field information
  	
	
  };
}

