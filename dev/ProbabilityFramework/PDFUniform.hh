// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLPROB_PDFUNIFORM_HEADER
#define RAVLPROB_PDFUNIFORM_HEADER 1
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Omni/Prob/PDFContinuousAbstract.hh"

namespace RavlProbN {
  using namespace RavlN;

  //! userlevel=Develop
  //: Probability distribution class for a continuous variable using a uniform function
  class PDFUniformBodyC
    : public PDFContinuousAbstractBodyC {
  public:
    PDFUniformBodyC(const RandomVariableContinuousC& variable, const RealRangeC& interval);
    //: Constructor
    //!param: variable - the random variable for the distribution
    //!param: interval - the interval over which the distribution != 0

    virtual ~PDFUniformBodyC();
    //: Destructor
    
    virtual RealT MeasureProbability(RealT value) const;
    //: Calculate the probability that the variable takes the specified value
    //!param: value - a value for the variable
    //!return: the probability that the variable takes the specified value

  private:
    void SetInterval(const RealRangeC& interval);
    //: Set the interval

  private:
    RealT m_probability;
    //: The uniform probability

    RealRangeC m_interval;
    //: The interval over which the probability != 0
  };

  //! userlevel=Normal
  //: Probability distribution class for a continuous variable using a uniform function
  //!cwiz:author
  
  class PDFUniformC
    : public PDFContinuousAbstractC
  {
  public:
    PDFUniformC(const RandomVariableContinuousC& variable, const RealRangeC& interval)
      : PDFContinuousAbstractC(new PDFUniformBodyC(variable, interval))
    {}
    //: Constructor
    //!param: variable - the random variable for the distribution
  };

}

#endif
