/*
 * Plot.hh
 *
 *  Created on: 10 Aug 2012
 *      Author: kieron
 */

#ifndef RAVLN_PLOT2D_HH_
#define RAVLN_PLOT2D_HH_

#include "Ravl/SmartPtr.hh"
#include "Ravl/String.hh"
#include "Ravl/Point2d.hh"
#include "Ravl/SArray1d.hh"
#include "Ravl/Image/Image.hh"
#include "Ravl/Image/ByteRGBValue.hh"
#include "Ravl/RealRange1d.hh"
#include "Ravl/PatternRec/DataSetVectorLabel.hh"

namespace RavlN {
  
  /*
   * Base class for plotting 2D points.  To do any plotting you will need
   * to use the GnuPlot2dC class.....
   */
  class Plot2dC : public RavlN::RCBodyVC
  {
  public:
    Plot2dC(const StringC & title);
    //: Construct with a set number of plots

    virtual bool Plot(const SArray1dC<Point2dC> & data, const StringC & dataName = "");
    //: Plot all graphs on single canvas

    virtual bool PlotFunction(const StringC & function);
    //: Plot a function, e.g. sin(x)

    virtual bool ScatterPlot(const DataSetVectorLabelC & dataSet, const IndexC & feature1 = 0, const IndexC & feature2 = 1);
    //: Make a scatter plot of the data

    virtual bool SetXLabel(const StringC & xlabel);
    //: Set the x-label

    virtual bool SetYLabel(const StringC & ylabel);
    //: Set the y-label

    virtual bool SetXRange(const RealRangeC & xrange);
    //: Set the range of the x-axis

    virtual bool SetYRange(const RealRangeC & yrange);
    //: Set the range of the y-axis

    bool SetAxisRange(const RealRangeC & xrange, const RealRangeC & yrange);
    //: Set axis range of first plot

    virtual bool SetLineStyle(const StringC & lineStyle);
    //: Set line style of plot

    virtual bool Command(const StringC & command);
    //: General method to send a command to the ploting library

    typedef RavlN::SmartPtrC<Plot2dC> RefT;

  protected:
    StringC m_title; //!< The overall title

  };

} /* namespace RavlN */
#endif /* PLOT2D_HH_ */
