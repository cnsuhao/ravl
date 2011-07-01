/*
 * FuncInterpolateGridGrid.hh
 *
 *  Created on: 24 Jun 2011
 *      Author: charles
 */

#ifndef RAVL_FuncInterpolateGridGRID_HH_
#define RAVL_FuncInterpolateGridGRID_HH_

#include "Ravl/PatternRec/FuncInterpolate.hh"
#include "Ravl/RealHistogramNd.hh"

namespace RavlN {

  //! userlevel=Develop
  //: Interpolated function.

  class FuncInterpolateGridBodyC
    : public FuncInterpolateBodyC
  {
  public:
    FuncInterpolateGridBodyC();
    //: Construct from a transform matrix.

    FuncInterpolateGridBodyC(const XMLFactoryContextC &factory);
    //: Factory constructor

    FuncInterpolateGridBodyC(const RealHistogramNdC<VectorC> &data,const SArray1dC<RealRangeC> &outputLimits);
    //: Constructor

    FuncInterpolateGridBodyC(istream &strm);
    //: Load from stream.

    FuncInterpolateGridBodyC(BinIStreamC &strm);
    //: Load from binary stream.

    virtual bool Save (ostream &out) const;
    //: Writes object to stream, can be loaded using constructor

    virtual bool Save (BinOStreamC &out) const;
    //: Writes object to stream, can be loaded using constructor

    virtual VectorC Apply(const VectorC &data) const;
    //: Apply transform to data.

  protected:
    RealHistogramNdC<VectorC> m_data;
  };


  //! userlevel=Normal
  //: Linear function.

  class FuncInterpolateGridC
    : public FuncInterpolateC
  {
  public:
    FuncInterpolateGridC()
    {}
    //: Default constructor.

    FuncInterpolateGridC(istream &strm);
    //: Load from stream.

    FuncInterpolateGridC(BinIStreamC &strm);
    //: Load from binary stream.

    FuncInterpolateGridC(const RealHistogramNdC<VectorC> &data,const SArray1dC<RealRangeC> &outputLimits)
     : FuncInterpolateC(new FuncInterpolateGridBodyC(data,outputLimits))
    {}
    //: Constructor

  protected:
    FuncInterpolateGridC(FuncInterpolateGridBodyC &bod)
      : FuncInterpolateC(bod)
    {}
    //: Body constructor.

    FuncInterpolateGridC(FuncInterpolateGridBodyC *bod)
      : FuncInterpolateC(bod)
    {}
    //: Body ptr constructor.

    FuncInterpolateGridBodyC &Body()
    { return static_cast<FuncInterpolateGridBodyC &>(FunctionC::Body()); }
    //: Access body.

    const FuncInterpolateGridBodyC &Body() const
    { return static_cast<const FuncInterpolateGridBodyC &>(FunctionC::Body()); }
    //: Access body.

  public:
  };

}



#endif /* FuncInterpolateGridGRID_HH_ */
