/*
 * Plot.hh
 *
 *  Created on: 10 Aug 2012
 *      Author: kieron
 */

#ifndef RAVLN_GNUPLOT2D_HH_
#define RAVLN_GNUPLOT2D_HH_

#include "Ravl/Plot/Plot2d.hh"
#include "Ravl/OS/ChildOSProcess.hh"

namespace RavlN {
  
  /*
   * Perform some plotting using 'gnuplot'
   */
  class GnuPlot2dC : public Plot2dC
  {
  public:
    GnuPlot2dC(const StringC & title);
    //: Construct with a set number of plots

    virtual bool Plot(const SArray1dC<Point2dC> & points, const StringC & dataName = "");
    //: Plot all graphs on single canvas

    virtual bool PlotFunction(const StringC & function);
    //: Plot a function

    virtual bool ScatterPlot(const DataSetVectorLabelC & dataSet, const IndexC & fv1 = 0, const IndexC & fv2 = 1);
    //: Make a scatter plot of the data.  Only the first two dimensions will be used...

    virtual bool SetXLabel(const StringC & xlabel);
    //: Set the x-label

    virtual bool SetYLabel(const StringC & ylabel);
    //: Set the y-label

    virtual bool SetXRange(const RealRangeC & xrange);
    //: Set the range of the x-axis

    virtual bool SetYRange(const RealRangeC & yrange);
    //: Set the range of the y-axis

    virtual bool SetLineStyle(const StringC & lineStyle);
    //: Set plot style

    virtual bool Command(const StringC & command);
    //: General method to send a command to the plotting library

    typedef RavlN::SmartPtrC<GnuPlot2dC> RefT;

  protected:
    void Flush();
    ChildOSProcessC m_gnuPlot; //< The GnuPlot process

  };

} /* namespace RavlN */
#endif /* PLOT2D_HH_ */
