// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html



%include "Ravl/Swig2/DesignFuncReduce.i"
%include "Ravl/Swig2/VectorMatrix.i"
%include "Ravl/Swig2/DataSetVectorLabel.i"

%{
#ifdef SWIGPERL
#undef Copy
#endif

#include "Ravl/PatternRec/DesignFuncLDA.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN {

	class DesignFuncLDAC
    : public DesignFuncReduceC
  {
  public:
    DesignFuncLDAC();
    //: Default constructor.
    // Creates an invalid handle.
    
    DesignFuncLDAC(RealT variationPreserved);
    //: Constructor 
    // "variationPreserved" is amount of variation to attempt to preserve in reduced set.
    // 0 -> None; 1 -> All; >1 (truncated to int) -> Size of set preserved.

    MatrixC &Lda();
    //: Access the transformation matrix.

    const MatrixC &Lda() const;
    //: Access the transformation matrix.
    
    VectorC &Mean();
    //: Access mean vector.
    
    const VectorC &Mean() const;
    //: Access mean vector.

    UIntT &SamplesPerClass();
    //: Access number of samples per class used for LDA training

    const UIntT &SamplesPerClass()const;
    //: Access number of samples per class used for LDA training

    FunctionC Apply(const DataSetVectorLabelC & data);
    //: Create dimensionality reduction function from the given labelled data sets.
    // Applies PCA for initial dimension reduction, then uses LDA.

    //inline FunctionC Apply(SampleStreamC<VectorC> &inPca,  SampleStream2C<VectorC, StringC> &inLda);
    //: Create dimensionality reduction function from the 2 streams.
    // This method uses streams so that you don't have to store all the data in memory.<br>
    //!param: inPca - uses this unlabelled stream to do some initial PCA dimension reduction (could be same stream as <code>inLda</code>)
    //!param: inLda - uses this labelled stream of vectors to do dimension reduction using LDA. Note, it is assumed that the labels are grouped together.

    //FunctionC Apply(SampleStreamC<TVectorC<float> > &inPca, SampleStream2C<TVectorC<float>, StringC> &inLda);
    //: Create function from the 2 streams.
    // This method uses streams so that you don't have to store all the data in memory.<br>
    //!param: inPca - uses this stream to do some initial PCA dimension reduction (could be same stream as inLda)
    //!param: inLda - uses this labelled stream of vectors to do dimension reduction using LDA

    //FunctionC Apply(SampleStreamVectorC & inPca, SampleStreamVectorLabelC &inLda);
    //: Create function from labelled sample stream of vectors.
    // This method uses streams so that you don't have to store all the data in memory.<br>
    //!param: inPca - data used to construct PCA.
    //!param: inLda - data used to construct LDA.
  
  };


}
