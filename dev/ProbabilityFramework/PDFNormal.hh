// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLPROB_PDFNORMAL_HEADER
#define RAVLPROB_PDFNORMAL_HEADER 1
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Omni/Prob/PDFContinuousAbstract.hh"

namespace RavlProbN {
  using namespace RavlN;

  //! userlevel=Develop
  //: Probability distribution class for a continuous variable using a gaussian function
  class PDFNormalBodyC
    : public PDFContinuousAbstractBodyC {
  public:
    PDFNormalBodyC(const RandomVariableContinuousC& variable, RealT mean, RealT variance);
    //: Constructor
    //!param: variable - the random variable for the distribution
    //!param: mean - the mean of the distribution
    //!param: variance - the variance of the distribution

    virtual ~PDFNormalBodyC();
    //: Destructor
    
    virtual RealT MeasureProbability(RealT value) const;
    //: Calculate the probability that the variable takes the specified value
    //!param: value - a value for the variable
    //!return: the probability that the variable takes the specified value

  private:
    void SetMeanAndVariance(RealT mean, RealT variance);
    //: Set the mean and variance

  private:
    RealT m_mean;
    //: The mean of the distribution

    RealT m_constant1;
    //: Cached constant from equation = 1.0 / (Sqrt (variance * 2pi));

    RealT m_constant2;
    //: Cached constant from equation = -1.0 / (2.0 * variance);
  };

  //! userlevel=Normal
  //: Probability distribution class for a continuous variable using a gaussian function
  //!cwiz:author
  
  class PDFNormalC
    : public PDFContinuousAbstractC
  {
  public:
    PDFNormalC(const RandomVariableContinuousC& variable, RealT mean, RealT variance)
      : PDFContinuousAbstractC(new PDFNormalBodyC(variable, mean, variance))
    {}
    //: Constructor
    //!param: variable - the random variable for the distribution
    //!param: mean - the mean of the distribution
    //!param: variance - the variance of the distribution
  };

}

#endif
