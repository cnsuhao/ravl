// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Omni/Prob/PDFDesignerFactory.hh"
#include "Ravl/OS/SysLog.hh"
#include "Omni/Prob/PDFContinuousDesignerNormal.hh"
#include "Omni/Prob/PDFContinuousDesignerUniform.hh"

#define DODEBUG 0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlProbN {
  using namespace RavlN;
  
  PDFDesignerFactoryBodyC::PDFDesignerFactoryBodyC() {
    m_continuousDesigners.InsLast(PDFContinuousDesignerNormalC::GetInstance());
    m_continuousDesigners.InsLast(PDFContinuousDesignerUniformC::GetInstance());
  }

  PDFDesignerFactoryBodyC::~PDFDesignerFactoryBodyC() {
  }

  PDFContinuousAbstractC PDFDesignerFactoryBodyC::DesignPDFContinuous(const RandomVariableContinuousC& variable, const DListC<RealT>& realSamples) const {
    PDFContinuousAbstractC pdf,bestPdf;
    RealT error,bestError = RavlConstN::maxReal;
    for (DLIterC<PDFContinuousDesignerC> it(m_continuousDesigners); it; it++) {
      pdf = it->CreatePDF(variable, realSamples);
      error = MeasureError(pdf, realSamples);
      if (!bestPdf.IsValid() || error < bestError) {
        bestPdf = pdf;
        bestError = error;
      }
    }
    return bestPdf;
  }

  RealT PDFDesignerFactoryBodyC::MeasureError(const PDFContinuousAbstractC& pdf, const DListC<RealT>& realSamples) const {
    return 0.0;
  }

  PDFDesignerFactoryC PDFDesignerFactoryC::m_instance;

  PDFDesignerFactoryC PDFDesignerFactoryC::GetInstance() {
    if (!m_instance.IsValid())
      m_instance = PDFDesignerFactoryC(true);
    return m_instance;
  }

}
