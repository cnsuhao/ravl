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
#include "Ravl/PatternRec/Function.hh"
#include "Ravl/LineABC2d.hh"

namespace RavlN {
  
  /*
   * Base class for plotting 2D points.  To do any plotting you will need
   * to use the GnuPlot2dC class.....
   */
  class Plot2dC : public RavlN::RCBodyVC
  {
  public:
    Plot2dC(const StringC & title = "My Plot");
    //: Construct with a set number of plots

    virtual bool Plot(const SArray1dC<Point2dC> & data, const StringC & dataName = "");
    //: Plot points, optional dataName will appear in the legend

    virtual bool Plot(const LineABC2dC & line);
    //: Plot a straight line

    virtual bool Plot(const StringC & function);
    //: Plot a function using a string, e.g. sin(x)

    virtual bool Plot(const DataSetVectorLabelC & dataSet, UIntT feature1 = 0, UIntT feature2 = 1, UIntT samplesPerClass = 0);
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
    //: Set line style of plot, points, line e.t.c.

    virtual bool Command(const StringC & command);
    //: General method to send a command to the plotting library, e.g. gnuplot

    typedef RavlN::SmartPtrC<Plot2dC> RefT;

  protected:
    StringC m_title; //!< The overall title


  };

} /* namespace RavlN */
#endif /* PLOT2D_HH_ */
