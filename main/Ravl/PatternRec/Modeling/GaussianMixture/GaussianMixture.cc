// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2002, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlPatternRec
//! author="Charles Galambos"
//! file="Ravl/PatternRec/DimensionReduction/FuncSubset.cc"

#include "Ravl/PatternRec/GaussianMixture.hh"
#include "Ravl/BinStream.hh"
#include "Ravl/SArray1dIter.hh"
#include "Ravl/SArray1dIter4.hh"
#include "Ravl/VirtualConstructor.hh"

namespace RavlN {
  
  //: Constructor from an array of indexes.
  
  GaussianMixtureBodyC::GaussianMixtureBodyC(const SArray1dC<MeanCovarianceC> & prms, const SArray1dC<RealT> & wgt, bool diag)
    : params(prms), weights(wgt), invCov(prms.Size()), det(prms.Size()), isDiagonal(diag)
  {
    //: Lets do some checks
    if(params.Size()!=weights.Size()) 
      RavlIssueError("Gaussian Mixture parameters not of same dimensionality as mixing weights");
    
    UIntT outputSize = params.Size();
    UIntT inputSize = params[0].Mean().Size();

    if(outputSize<1) 
      RavlIssueError("Gaussian Mixture model has output dimension of < 1");

    if(inputSize<1) 
      RavlIssueError("Gaussian Mixture model has input dimension < 1");

    OutputSize(outputSize);
    InputSize(inputSize);

    //: For speed, lets precompute some stuff
    konst =  Pow(2.0* RavlConstN::pi, (RealT)inputSize/2.0);
    
    //: First we have to check for very small variances
    //: The smallest variance we allow
    RealT smallVariance = 0.001;
    RealT smallDeterminant = 1e-20;

#if 0
    //: lets regularise our model
    for(SArray1dIterC<MeanCovarianceC> paramIt(params);paramIt;paramIt++)  {
      for (UIntT i=0; i<inputSize; i++)
	paramIt.Data().Covariance()[i][i] += smallVariance;
    }
#endif

    //: compute inverse and determinant of models
    for(SArray1dIterC<MeanCovarianceC>paramIt(params);paramIt;paramIt++)  {
      IndexC i = paramIt.Index();
      if(!isDiagonal) {
	invCov[i] = paramIt.Data().Covariance().NearSingularInverse(det[i]);             

      } else {
	det[i]=1.0;
	invCov [i] = MatrixRSC(inputSize);
	invCov[i].Fill(0.0);
	for(UIntT j=0;j<inputSize;j++) {
	  invCov[i][j][j]=1.0/paramIt.Data().Covariance()[j][j];
	  det[i] *= paramIt.Data().Covariance()[j][j]; 
	}
      }
      
      if(det[i]<smallDeterminant) {
	//: if this is called then we have a problem on one component
	//: having a very small variance.  We do try and avoid this
	//: by setting a minimum allowed variance, but it is not foolproof.
	RavlIssueError("The deteminant is too small (or negative).  Unsuitable data, perhaps use PCA.");
      } 
    }
    
  }

  //: Load from stream.
  GaussianMixtureBodyC::GaussianMixtureBodyC(istream &strm) 
    : FunctionBodyC(strm) { 
    strm >> params >> weights >> invCov >> det >> konst >> isDiagonal; 
  }
  
  
  //: Writes object to stream.  
  bool GaussianMixtureBodyC::Save (ostream &out) const {
    if(!FunctionBodyC::Save(out))
      return false;
    out << '\n' << params << '\n' << weights << '\n' << invCov << '\n' << det << '\n' << konst << '\n' << isDiagonal;
    return true;    
  }
  
  //: Load from binary stream.  
  GaussianMixtureBodyC::GaussianMixtureBodyC(BinIStreamC &strm) 
    : FunctionBodyC(strm) { 
    strm >> params  >> weights >> invCov >> det >> konst >> isDiagonal; 
    OutputSize(params.Size());
    InputSize(params[0].Mean().Size());
    
  }

  //: Writes object to binary stream.  
  bool GaussianMixtureBodyC::Save (BinOStreamC &out) const {
    if(!FunctionBodyC::Save(out))
      return false;
    out << params << weights << invCov << det << konst << isDiagonal;
    return true;
  }
  
  //: Compute the density at a given point 
  VectorC GaussianMixtureBodyC::Apply(const VectorC & data) const  {
    
    //: do some checks
    if(data.Size()!=InputSize()) 
      RavlIssueError("Input data of different dimension to that of model");
    
    VectorC out(OutputSize());
    for(SArray1dIter4C<MeanCovarianceC, RealT, MatrixRSC, RealT>it(params, weights, invCov, det);it;it++) {
      VectorC D = data - it.Data1().Mean();
      out[it.Index()]  = it.Data2() * ((1.0/(konst * Sqrt(it.Data4()))) * Exp(-0.5 * D.Dot(( it.Data3() * D))));
    }

    return out;
  }

  RAVL_INITVIRTUALCONSTRUCTOR_FULL(GaussianMixtureBodyC,GaussianMixtureC,FunctionC);

}
