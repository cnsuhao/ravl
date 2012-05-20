// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2002, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! lib=RavlPatternRec
//! file="Ravl/PatternRec/Classify/ClassifierLogisticRegression.cc"

#include "Ravl/PatternRec/ClassifierLogisticRegression.hh"
#include "Ravl/VirtualConstructor.hh"
#include "Ravl/BinStream.hh"

#define DODEBUG 1
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlN {
  
  //: Create classifier from function.
  
  ClassifierLogisticRegressionBodyC::ClassifierLogisticRegressionBodyC(const FunctionC &nfunc,const MatrixC &weights)
    : ClassifierBodyC(weights.Rows()),
      m_norm(nfunc),
      m_weights(weights)
  {}
  
  //: Load from stream.
  
  ClassifierLogisticRegressionBodyC::ClassifierLogisticRegressionBodyC(istream &strm)
    : ClassifierBodyC(strm)
  { 
    IntT version;
    strm >> version;
    if(version != 1)
      throw ExceptionOutOfRangeC("ClassifierLogisticRegressionBodyC::ClassifierLogisticRegressionBodyC(istream &), Unrecognised version number in stream. ");

    strm >> m_norm >> m_weights;
  }
  
  //: Load from binary stream.
  
  ClassifierLogisticRegressionBodyC::ClassifierLogisticRegressionBodyC(BinIStreamC &strm)
    : ClassifierBodyC(strm)
  {
    IntT version;
    strm >> version;
    if(version != 1)
      throw ExceptionOutOfRangeC("ClassifierLogisticRegressionBodyC::ClassifierLogisticRegressionBodyC(BinIStreamC &), Unrecognised version number in stream. ");
    strm >> m_norm >> m_weights;
  }
  
  //: Writes object to stream, can be loaded using constructor
  
  bool ClassifierLogisticRegressionBodyC::Save(ostream &out) const {
    if(!ClassifierBodyC::Save(out))
      return false;
    IntT version = 1;
    out << ' ' << version << ' ' << m_norm << m_weights;
    return true;
  }
  
  //: Writes object to stream, can be loaded using constructor
  
  bool ClassifierLogisticRegressionBodyC::Save(BinOStreamC &out) const {
    if(!ClassifierBodyC::Save(out))
      return false;
    IntT version = 1;
    out << version << m_norm << m_weights;
    return true;    
  }
  
  static VectorC MakeBias()
  {
    VectorC bias(1);
    bias[0] = 1;
    return bias;
  }

  static const VectorC &BiasVector()
  {
    static VectorC bias = MakeBias();
    return bias;
  }

  //: Classifier vector 'data' return the most likely label.
  
  UIntT ClassifierLogisticRegressionBodyC::Classify(const VectorC &data) const {
    VectorC vec;
    if(m_norm.IsValid())
      vec = m_norm(data);
    else
      vec = data;
    VectorC result = m_weights * VectorC(BiasVector().Join(vec));
    return result.MaxIndex().V();
  }
  
  //: Estimate the confidence for each label.
  // The meaning of the confidence assigned to each label depends
  // on the classifier used. The higher the confidence the more likely
  // it is the label is correct.
  
  VectorC ClassifierLogisticRegressionBodyC::Confidence(const VectorC &data) const {
    VectorC vec;
    if(m_norm.IsValid())
      vec = m_norm(data);
    else
      vec = data;
    VectorC result = Sigmoid(m_weights * vec);
    return result.MakeUnit();
  }
  
  RAVL_INITVIRTUALCONSTRUCTOR_FULL(ClassifierLogisticRegressionBodyC,ClassifierLogisticRegressionC,ClassifierC);
  
  
}
