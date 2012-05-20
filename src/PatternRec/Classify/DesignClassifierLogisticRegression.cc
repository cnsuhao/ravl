// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2012, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! lib=RavlPatternRec
//! file="Ravl/PatternRec/Classify/DesignClassifierLogisticRegression.cc"

#include "Ravl/PatternRec/DesignClassifierLogisticRegression.hh"
#include "Ravl/VirtualConstructor.hh"
#include "Ravl/BinStream.hh"
#include "Ravl/XMLFactoryRegister.hh"
#include "Ravl/PatternRec/Cost.hh"
#include "Ravl/PatternRec/DataSet2.hh"
#include "Ravl/PatternRec/DataSet3.hh"
#include "Ravl/PatternRec/DataSet2Iter.hh"
#include "Ravl/PatternRec/DataSet3Iter.hh"
#include "Ravl/PatternRec/SampleVector.hh"
#include "Ravl/PatternRec/OptimiseConjugateGradient.hh"
#include "Ravl/PatternRec/OptimiseDescent.hh"
#include "Ravl/PatternRec/SampleLabel.hh"
#include "Ravl/SysLog.hh"
#include "Ravl/MeanCovariance.hh"
#include "Ravl/PatternRec/ClassifierLogisticRegression.hh"
#include "Ravl/SysLog.hh"

#define DODEBUG 0
#if DODEBUG
#define ONDEBUG(x) x
#include "Ravl/StrStream.hh"
#include "Ravl/OS/Date.hh"
#else
#define ONDEBUG(x)
#endif

namespace RavlN {

  //! Cost function for logistic regression.

  class CostLogisticRegressionC
   : public CostBodyC
  {
  public:
    //! Constructor
    CostLogisticRegressionC(UIntT label,UIntT vecSize,const SampleC<VectorC> &in,const SampleC<UIntT> &out)
     : m_label(label),
       m_in(in),
       m_out(out)
    {
      ParametersC parameters(vecSize,true);
      SetParameters(parameters);
      RavlAssert(parameters.StartX().Size() == vecSize);
    }

    //! Compute the cost of a solution.
    virtual RealT Cost (const VectorC &X) const;

    //! Compute the jacobian.
    virtual VectorC Jacobian1(const VectorC &X) const;

  protected:
    UIntT m_label;
    SampleC<VectorC> m_in;
    SampleC<UIntT> m_out;
    SampleC<RealT> m_weight;
  };


  //! Compute the cost of a solution.

  RealT CostLogisticRegressionC::Cost (const VectorC &theta) const
  {
    RealT cost = 0;
    //ONDEBUG(RavlDebug("Theta:%s",RavlN::StringOf(theta).c_str()));
    for(DataSet2IterC<SampleC<VectorC>,SampleC<UIntT> > it(m_in,m_out);it;it++)
    {
      //ONDEBUG(RavlDebug("Data:%s Theta:%s ",RavlN::StringOf(it.Data1()).c_str(),RavlN::StringOf(theta).c_str()));
      RealT dotProdS = Sigmoid(it.Data1().Dot(theta));
      //RavlDebug("Dot %f ",dotProdS);
      if(m_label == it.Data2()) {
        cost += -Log(dotProdS);
      } else {
        cost += -Log(1-dotProdS);
      }
    }

    //J = sum(-y .* log( sigmoid(X * theta) ) - (1 -y) .* log(1 - sigmoid(X * theta)))/ m;
    RealT fcost = cost / ((RealT) m_in.Size());
    //ONDEBUG(RavlDebug("Cost=%f",fcost));
    //RavlN::Sleep(1.0);
    return fcost;
  }

  //! Compute the jacobian.

  VectorC CostLogisticRegressionC::Jacobian1(const VectorC &theta) const
  {
    VectorC grad(theta.Size());
    grad.Fill(0);
    for(DataSet2IterC<SampleC<VectorC>,SampleC<UIntT> > it(m_in,m_out);it;it++)
    {
      RealT num = Sigmoid(it.Data1().Dot(theta));
      if(it.Data2() == m_label) {
        num -= 1.0;
      }
      for(unsigned i = 0;i < grad.Size();i++)
        grad[i] += it.Data1()[i] * num;
    }
    //grad = (1 / m) * X' * (sigmoid(X * theta) - y) ;
    grad /= ((RealT) m_in.Size());
    //ONDEBUG(RavlDebug("Grad @ %s = %s",RavlN::StringOf(theta).c_str(),RavlN::StringOf(grad).c_str()));
    return grad;
  }

  // -------------------------------------------------------------------------

  //: Constructor.
  
  DesignClassifierLogisticRegressionBodyC::DesignClassifierLogisticRegressionBodyC(RealT regularisation,const OptimiseC &optimiser)
   : m_optimiser(optimiser),
     m_regularisation(regularisation)

  {
    if(!m_optimiser.IsValid()) {
      //m_optimiser = OptimiseConjugateGradientC(1000);
      m_optimiser = OptimiseDescentC(1000,1e-3);
    }
  }
  
  //: Load from XML factory
  
  DesignClassifierLogisticRegressionBodyC::DesignClassifierLogisticRegressionBodyC(const XMLFactoryContextC & factory)
    : DesignClassifierSupervisedBodyC(factory),
      m_regularisation(factory.AttributeReal("regularisation",0))
  {
    if(!factory.UseChildComponent("Optimiser",m_optimiser)) {
      m_optimiser = OptimiseConjugateGradientC(1000);
    }
  }
  
  //: Load from stream.
  
  DesignClassifierLogisticRegressionBodyC::DesignClassifierLogisticRegressionBodyC(istream &strm)
    : DesignClassifierSupervisedBodyC(strm)
  {
    int version;
    strm >> version;
    if(version != 0)
      throw ExceptionOutOfRangeC("DesignClassifierLogisticRegressionBodyC::DesignClassifierLogisticRegressionBodyC(istream &), Unrecognised version number in stream. ");
    //strm >> k >> distanceMetric >> useAverageKNN;
  }
  
  //: Load from binary stream.
  
  DesignClassifierLogisticRegressionBodyC::DesignClassifierLogisticRegressionBodyC(BinIStreamC &strm)
    : DesignClassifierSupervisedBodyC(strm)
  {
    int version;
    strm >> version;
    if(version != 0)
      throw ExceptionOutOfRangeC("DesignClassifierLogisticRegressionBodyC::DesignClassifierLogisticRegressionBodyC(BinIStreamC &), Unrecognised version number in stream. ");
   strm >> m_optimiser;
  }
  
  //: Writes object to stream, can be loaded using constructor
  
  bool DesignClassifierLogisticRegressionBodyC::Save (ostream &out) const {
    if(!DesignClassifierSupervisedBodyC::Save(out))
      return false;
    int version = 0;
    out << ' ' << version;
    return true;
  }
  
  //: Writes object to stream, can be loaded using constructor
  
  bool DesignClassifierLogisticRegressionBodyC::Save (BinOStreamC &out) const {
    if(!DesignClassifierSupervisedBodyC::Save(out))
      return false;
    int version = 0;
    out << version << m_optimiser;
    return true;
  }
  
  //: Create a classifier.
  
  ClassifierC DesignClassifierLogisticRegressionBodyC::Apply(const SampleC<VectorC> &in,const SampleC<UIntT> &out)
  {
    RavlAssertMsg(in.Size() == out.Size(),"DesignClassifierLogisticRegressionBodyC::Apply(), Sample of vector and labels should be the same size.");
    ONDEBUG(RavlDebug("Designing logistic regression classifier."));

    SampleVectorC inVec(in);
    MeanCovarianceC meanCov = inVec.MeanCovariance();
    SampleVectorC normVec(in.Size());

    SampleLabelC labels(out);

    inVec.Normalise(meanCov,normVec,true);
    unsigned features = normVec.VectorSize();
    VectorC theta(features);
    theta.Fill(0);

    unsigned maxLabel = labels.MaxValue() ;
    ONDEBUG(RavlDebug("Features:%u Labels:%u",features,maxLabel));

    MatrixC weights(maxLabel+1,features);
    // If its a two class problem we only need 1 classifier.
    if(maxLabel == 2) maxLabel = 1;
    for(unsigned i = 0;i <= maxLabel;i++) {
      ONDEBUG(RavlDebug("Processing column %i ",i));
      CostC costFunc(new CostLogisticRegressionC(i,features,normVec,labels));
      ONDEBUG(RavlDebug("Start at:%s",RavlN::StringOf(costFunc.StartX()).c_str()));

      RealT minimumCost;
      VectorC result = m_optimiser.MinimalX (costFunc,minimumCost);
      weights.SetRow(i,result);
    }

    FunctionC normFunc = inVec.NormalisationFunction(meanCov);
    return ClassifierLogisticRegressionC(normFunc,weights);
  }
  
  //: Create a classifier with weights for the samples.
  
  ClassifierC DesignClassifierLogisticRegressionBodyC::Apply(const SampleC<VectorC> &in,
						  const SampleC<UIntT> &out,
						  const SampleC<RealT> &weight)
  {
    RavlAssertMsg(0,"DesignClassifierLogisticRegressionBodyC::Apply(in,out,weight), Not implemented. Send a feature request! ");
    return ClassifierC();
  }
 
  //////////////////////////////////////////////////////////
  
  RavlN::XMLFactoryRegisterHandleConvertC<DesignClassifierLogisticRegressionC, DesignClassifierSupervisedC> g_registerXMLFactoryDesignClassifierLogisticRegression("RavlN::DesignClassifierLogisticRegressionC");

  RAVL_INITVIRTUALCONSTRUCTOR_FULL(DesignClassifierLogisticRegressionBodyC,DesignClassifierLogisticRegressionC,DesignClassifierSupervisedC);

  void linkDesignClassifierLogisticRegression()
  {}

}
