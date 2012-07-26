// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2011, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! lib=RavlPatternRec
//! author="Charles Galambos"
//! file="Ravl/PatternRec/Modeling/Basic/DesignFuncInterpolateGrid.cc"

#include "Ravl/PatternRec/DesignFuncInterpolateGrid.hh"
#include "Ravl/PatternRec/SampleVector.hh"
#include "Ravl/PatternRec/SampleIter.hh"
#include "Ravl/PatternRec/FuncInterpolateGrid.hh"
#include "Ravl/MatrixRUT.hh"
#include "Ravl/MatrixRS.hh"
#include "Ravl/BinStream.hh"
#include "Ravl/VirtualConstructor.hh"
#include "Ravl/Exception.hh"

#define DODEBUG 1
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlN {

  //: Create least squares designer.

  DesignFuncInterpolateGridBodyC::DesignFuncInterpolateGridBodyC()
  {}

  //: Load from stream.

  DesignFuncInterpolateGridBodyC::DesignFuncInterpolateGridBodyC(std::istream &strm)
    : DesignFunctionSupervisedBodyC(strm)
  {
    int ver;
    strm >> ver;
    if(ver != 1)
      throw ExceptionUnexpectedVersionInStreamC("DesignFuncLSQBodyC::DesignFuncLSQBodyC(std::istream&), Unknown format version. ");
  }

  //: Load from binary stream.

  DesignFuncInterpolateGridBodyC::DesignFuncInterpolateGridBodyC(BinIStreamC &strm)
    : DesignFunctionSupervisedBodyC(strm)
  {
    ByteT ver;
    strm >> ver;
    if(ver != 1)
      throw ExceptionUnexpectedVersionInStreamC("DesignFuncInterpolateGridBodyC(BinIStreamC&), Unknown format version. ");
  }

  //: Writes object to stream, can be loaded using constructor

  bool DesignFuncInterpolateGridBodyC::Save (std::ostream &out) const {
    if(!DesignFunctionSupervisedBodyC::Save(out))
      return false;
    ByteT ver = 1;
    out << ((int) ver);
    return true;
  }

  //: Writes object to stream, can be loaded using constructor

  bool DesignFuncInterpolateGridBodyC::Save (BinOStreamC &out) const {
    if(!DesignFunctionSupervisedBodyC::Save(out))
      return false;
    char ver = 1;
    out << ver;
    return true;
  }

  //: Create function from the given data.

  FunctionC DesignFuncInterpolateGridBodyC::Apply(const SampleC<VectorC> &in,const SampleC<VectorC> &out)
  {
    size_t outputSize = 0;
    RealHistogramNdC<MeanNdC> stats(m_inputRange, m_gridSize);
    SArray1dC<MeanNdC> bins = stats.RawBins();
    for(unsigned i = 0;i < bins.Size();i++) {
      bins[i] = MeanNdC(outputSize);
    }

    return FunctionC();
  }

  //: Create function from the given data, and sample weights.

  FunctionC DesignFuncInterpolateGridBodyC::Apply(const SampleC<VectorC> &in,const SampleC<VectorC> &out,const SampleC<RealT> &weight)
  {
    return FunctionC();
  }

  //: Create function from the given data.
  // Note: Construction from a sample stream may not be implemented by all designers.

  FunctionC DesignFuncInterpolateGridBodyC::Apply(SampleStream2C<VectorC,VectorC > &in)
  {
    return FunctionC();
  }


  ////////////////////////////////////////////////////////////////////////

  RAVL_INITVIRTUALCONSTRUCTOR_FULL(DesignFuncInterpolateGridBodyC,DesignFuncInterpolateGridC,DesignFunctionSupervisedC);

}
