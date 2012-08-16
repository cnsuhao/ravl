// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html


%include "Ravl/Swig2/String.i"
%include "Ravl/Swig2/Point2d.i"
%include "Ravl/Swig2/Index.i"
%include "Ravl/Swig2/RealRange.i"
%include "Ravl/Swig2/SArray1d.i"

%{
#ifdef SWIGPERL
#undef Copy
#endif

#include "Ravl/Plot/GnuPlot2d.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN {

  class GnuPlot2dC {
  public:
    GnuPlot2dC(const StringC & title = "My Plot");
    //: Construct with a title

    bool Plot(const SArray1dC<Point2dC> & points, const StringC & dataName = "");
    //: Plot all graphs on single canvas

    bool PlotFunction(const StringC & function);
    //: Plot a function
 
 	bool PlotFunction(const FunctionC & function);
 
    bool ScatterPlot(const DataSetVectorLabelC & dataSet, const IndexC & fv1 = 0, const IndexC & fv2 = 1);
    //: Make a scatter plot of the data.  Only the first two dimensions will be used...

    bool SetXLabel(const StringC & xlabel);
    //: Set the x-label

    bool SetYLabel(const StringC & ylabel);
    //: Set the y-label

    bool SetXRange(const RealRangeC & xrange);
    //: Set the range of the x-axis

    bool SetYRange(const RealRangeC & yrange);
    //: Set the range of the y-axis

    bool SetLineStyle(const StringC & lineStyle);
    //: Set plot style

    bool Command(const StringC & command);
    //: General method to send a command to the plotting library

  };
}
