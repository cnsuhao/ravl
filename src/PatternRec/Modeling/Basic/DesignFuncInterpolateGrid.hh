// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2011, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_DESIGNFUNCINTERPOLATEDGRID_HEADER
#define RAVL_DESIGNFUNCINTERPOLATEDGRID_HEADER 1
//! lib=RavlPatternRec
//! author="Charles Galambos"
//! docentry="Ravl.API.Pattern Recognition.Numerical Modeling"
//! file="Ravl/PatternRec/Modeling/Basic/DesignFuncInterpolateGrid.hh"

#include "Ravl/PatternRec/DesignFunctionSupervised.hh"
#include "Ravl/PatternRec/FuncLinearCoeff.hh"

namespace RavlN {
  class MatrixRUTC;

  //! userlevel=Develop
  //: Design function with Least Squares Fitting.

  class DesignFuncInterpolateGridBodyC
    : public DesignFunctionSupervisedBodyC
  {
  public:
    DesignFuncInterpolateGridBodyC()
    {}
    //: Default constructor.

    DesignFuncInterpolateGridBodyC(istream &strm);
    //: Load from stream.

    DesignFuncInterpolateGridBodyC(BinIStreamC &strm);
    //: Load from binary stream.

    virtual bool Save (ostream &out) const;
    //: Writes object to stream, can be loaded using constructor

    virtual bool Save (BinOStreamC &out) const;
    //: Writes object to stream, can be loaded using constructor

    DesignFuncInterpolateGridBodyC(UIntT norder,bool northogonal);
    //: Create least squares designer.

    virtual FuncLinearCoeffC CreateFunc(UIntT nin,UIntT nout);
    //: Create new function.

    virtual FunctionC Apply(const SampleC<VectorC> &in,const SampleC<VectorC> &out);
    //: Create function from the given data.

    virtual FunctionC Apply(const SampleC<VectorC> &in,const SampleC<VectorC> &out,const SampleC<RealT> &weight);
    //: Create function from the given data, and sample weights.

    virtual FunctionC Apply(SampleStream2C<VectorC,VectorC > &in);
    //: Create function from the given data.
    // Note: Construction from a sample stream may not be implemented by all designers.

    SArray1dC<IntT> FindCorrelatedParameters(const MatrixRUTC &mat,RealT thresh = 1e-6);
    //: Find correlated parameters.
    // The array contains -1 if the parameter is independent or the number of the
    // Parameter it correlates with.

  protected:
    SArray1dC<size_t> m_gridSize;
    SArray1dC<RealRangeC> m_inputRange;
  };

  //! userlevel=Normal
  //: Design function with Least Squares Fitting.

  class DesignFuncInterpolateGridC
    : public DesignFunctionSupervisedC
  {
  public:
    DesignFuncInterpolateGridC()
    {}
    //: Default constructor.
    // Creates an invalid handle.

    DesignFuncInterpolateGridC(istream &strm);
    //: Load from stream.

    DesignFuncInterpolateGridC(BinIStreamC &strm);
    //: Load from binary stream.

    DesignFuncInterpolateGridC(UIntT order,bool orthogonal)
      : DesignFunctionSupervisedC(*new DesignFuncInterpolateGridBodyC(order,orthogonal))
    {}
    //: Create designer.

  protected:
    DesignFuncInterpolateGridC(DesignFuncInterpolateGridBodyC &bod)
      : DesignFunctionSupervisedC(bod)
    {}
    //: Body constructor.

    DesignFuncInterpolateGridC(DesignFuncInterpolateGridBodyC *bod)
      : DesignFunctionSupervisedC(bod)
    {}
    //: Body ptr constructor.

  };

  inline istream &operator>>(istream &strm,DesignFuncInterpolateGridC &obj) {
    obj = DesignFuncInterpolateGridC(strm);
    return strm;
  }
  //: Load from a stream.
  // Uses virtual constructor.

  inline ostream &operator<<(ostream &out,const DesignFuncInterpolateGridC &obj) {
    obj.Save(out);
    return out;
  }
  //: Save to a stream.
  // Uses virtual constructor.

  inline BinIStreamC &operator>>(BinIStreamC &strm,DesignFuncInterpolateGridC &obj) {
    obj = DesignFuncInterpolateGridC(strm);
    return strm;
  }
  //: Load from a binary stream.
  // Uses virtual constructor.

  inline BinOStreamC &operator<<(BinOStreamC &out,const DesignFuncInterpolateGridC &obj) {
    obj.Save(out);
    return out;
  }
  //: Save to a stream.
  // Uses virtual constructor.

}


#endif
