/*
 * Plot.cc
 *
 *  Created on: 10 Aug 2012
 *      Author: kieron
 */

#include "Ravl/Plot/Plot2d.hh"
#include "Ravl/SArray1dIter.hh"
#include "Ravl/Assert.hh"

namespace RavlN {
  
  /*
   * Construct
   */
  Plot2dC::Plot2dC(const StringC & title) :
      m_title(title)
  {
  }

  bool Plot2dC::Plot(const SArray1dC<Point2dC> & data, const StringC & dataName)
  {
    RavlAssertMsg(0, "Abstract method called!");
    return false;
  }

  /*
   * Plot a function
   */
  bool Plot2dC::PlotFunction(const StringC & function)
  {
    RavlAssertMsg(0, "Abstract method called!");
    return false;
  }

  /*
   * Make a scatter plot of a data set
   */
  bool Plot2dC::ScatterPlot(const DataSetVectorLabelC & dataSet, const IndexC & fv1, const IndexC & fv2)
  {
    RavlAssertMsg(0, "Abstract method called!");
    return false;
  }

  /*
   * Set the X label
   */
  bool Plot2dC::SetXLabel(const StringC & xlabel)
  {
    RavlAssertMsg(0, "Abstract method called!");
    return false;
  }

  /*
   * Set the Y label
   */
  bool Plot2dC::SetYLabel(const StringC & ylabel)
  {
    RavlAssertMsg(0, "Abstract method called!");
    return false;
  }

  /*
   *  Set axis range of a plot
   */

  bool Plot2dC::SetXRange(const RealRangeC & xrange)
  {
    RavlAssertMsg(0, "Abstract method called!");
    return false;
  }

  /*
   *  Set axis range of a plot
   */

  bool Plot2dC::SetYRange(const RealRangeC & yrange)
  {
    RavlAssertMsg(0, "Abstract method called!");
    return false;
  }

  /*
   *  Set axis range of a plot
   */

  bool Plot2dC::SetAxisRange(const RealRangeC & xrange, const RealRangeC & yrange)
  {
    return SetXRange(xrange) && SetYRange(yrange);
  }
  //: Set axis range

  /*
   * Set the line style
   */

  bool Plot2dC::SetLineStyle(const StringC & lineStyle)
  {
    RavlAssertMsg(0, "Abstract method called!");
    return false;
  }

  /*
   * Send the plotting device a generic command
   */

  bool Plot2dC::Command(const StringC & command)
  {
    RavlAssertMsg(0, "Abstract method called!");
    return false;
  }
//: General method to send a command to the ploting library

} /* namespace RavlN */
