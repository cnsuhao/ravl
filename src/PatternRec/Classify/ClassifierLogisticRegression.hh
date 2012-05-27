// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2002, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_CLASSIFYDISCRIMINANTFUNCTION_HEADER
#define RAVL_CLASSIFYDISCRIMINANTFUNCTION_HEADER 1
///////////////////////////////////////////////////////////////////////
//! rcsid="$Id$"
//! author="Charles Galambos"
//! docentry="Ravl.API.Pattern Recognition.Classifier"
//! lib=RavlPatternRec
//! file="Ravl/PatternRec/Classify/ClassifierLogisticRegression.hh"

#include "Ravl/PatternRec/Classifier.hh"
#include "Ravl/PatternRec/Function.hh"

namespace RavlN {
  
  //! userlevel=Develop
  //: A classifier which does logistic regression
  
  class ClassifierLogisticRegressionBodyC
    : public ClassifierBodyC
  {
  public:
    ClassifierLogisticRegressionBodyC(const FunctionC &normFunc,const MatrixC &weights,bool prependUnit);
    //: Create classifier from a normalisation function and some weights.
    
    ClassifierLogisticRegressionBodyC(istream &strm);
    //: Load from stream.
    
    ClassifierLogisticRegressionBodyC(BinIStreamC &strm);
    //: Load from binary stream.
    
    virtual bool Save (ostream &out) const;
    //: Writes object to stream, can be loaded using constructor
    
    virtual bool Save (BinOStreamC &out) const;
    //: Writes object to stream, can be loaded using constructor
    
    virtual UIntT Classify(const VectorC &data) const;
    //: Classifier vector 'data' return the most likely label.
    
    virtual VectorC Confidence(const VectorC &data) const;
    //: Estimate the confidence for each label.
    // The meaning of the confidence assigned to each label depends
    // on the function used. The higher the confidence the more likely
    // it is the label is correct. The sum of the vector has been normalised
    // to 1.

  protected:
    FunctionC m_norm;
    MatrixC m_weights;
    bool m_prependUnit;
  };
  
  //! userlevel=Develop
  //: A classifier which does logistic regression

  class ClassifierLogisticRegressionC
    : public ClassifierC
  {
  public:
    ClassifierLogisticRegressionC()
    {}
    //: Default constructor.
    // Creates an invalid handle.
    
    ClassifierLogisticRegressionC(const FunctionC &normFunc,const MatrixC &weights,bool prependUnit)
      : ClassifierC(*new ClassifierLogisticRegressionBodyC(normFunc,weights,prependUnit))
    {}
    //: Create classifier from a normalisation function and some weights.
    
    ClassifierLogisticRegressionC(istream &strm);
    //: Load from stream.
    
    ClassifierLogisticRegressionC(BinIStreamC &strm);
    //: Load from binary stream.
    
  protected:
    ClassifierLogisticRegressionC(ClassifierLogisticRegressionBodyC &bod)
      : ClassifierC(bod)
    {}
    //: Body constructor.

    ClassifierLogisticRegressionC(ClassifierLogisticRegressionBodyC *bod)
      : ClassifierC(bod)
    {}
    //: Body constructor.
    
    ClassifierLogisticRegressionBodyC &Body()
    { return static_cast<ClassifierLogisticRegressionBodyC &>(ClassifierC::Body()); }
    //: Access body.

    const ClassifierLogisticRegressionBodyC &Body() const
    { return static_cast<const ClassifierLogisticRegressionBodyC &>(ClassifierC::Body()); }
    //: Access body.
  public:
  };
  
  inline istream &operator>>(istream &strm,ClassifierLogisticRegressionC &obj) {
    obj = ClassifierLogisticRegressionC(strm);
    return strm;
  }
  //: Load from a stream.
  // Uses virtual constructor.
  
  inline ostream &operator<<(ostream &out,const ClassifierLogisticRegressionC &obj) {
    obj.Save(out);
    return out;
  }
  //: Save to a stream.
  // Uses virtual constructor.
  
  inline BinIStreamC &operator>>(BinIStreamC &strm,ClassifierLogisticRegressionC &obj) {
    obj = ClassifierLogisticRegressionC(strm);
    return strm;
  }
  //: Load from a binary stream.
  // Uses virtual constructor.
  
  inline BinOStreamC &operator<<(BinOStreamC &out,const ClassifierLogisticRegressionC &obj) {
    obj.Save(out);
    return out;
  }
  //: Save to a stream.
  // Uses virtual constructor.
  
}

#endif
