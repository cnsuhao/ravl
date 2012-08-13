/*
 * Plot.cc
 *
 *  Created on: 10 Aug 2012
 *      Author: kieron
 */

#include "Ravl/Plot/GnuPlot2d.hh"
#include "Ravl/SArray1dIter.hh"
#include "Ravl/Assert.hh"
#include "Ravl/SysLog.hh"
#include "Ravl/Exception.hh"
#include "Ravl/PatternRec/DataSet2Iter.hh"
#include "Ravl/SArray1dIter2.hh"

namespace RavlN {
  
  /*
   * Construct
   */
  GnuPlot2dC::GnuPlot2dC(const StringC & title) :
      Plot2dC(title), m_gnuPlot((StringC) "gnuplot", false, false, true)
  {
    // check it is still running
    if (!m_gnuPlot.IsRunning()) {
      RavlError("GnuPlot not running!");
      throw ExceptionBadConfigC("gnuplot not running!");
    }

    // set-up the default terminal
    if (!title.IsEmpty()) {
      Command("set title \"" + title + "\"");
    }

    Command("set terminal wxt persist raise");

  }

  /*
   * Plot all on a single graph
   */

  bool GnuPlot2dC::Plot(const SArray1dC<Point2dC> & data, const StringC & dataName)
  {
    // check it is still running
    if (!m_gnuPlot.IsRunning()) {
      RavlError("GnuPlot not running!");
      return false;
    }

    // make the command
    StringC cmd;
    if (dataName.IsEmpty()) {
      cmd.form("plot \"-\" notitle");
    } else {
      cmd.form("plot \"-\" title \'%s\'", dataName.data());
    }
    Command(cmd);

    // copy the data across
    for (SArray1dIterC<Point2dC> it(data); it; it++) {
      StringC d;
      d.form("%f %f", it.Data().Row(), it.Data().Col());
      Command(d);
    }
    Command("end");
    Flush();

    return true;
  }


  /*
   * Plot a function
   */
  bool GnuPlot2dC::PlotFunction(const StringC & function)
  {

    m_gnuPlot.StdIn() << "plot " << function << endl;
    Flush();

    return true;
  }

  /*
   * Make a scatter plot of a data set
   */
  bool GnuPlot2dC::ScatterPlot(const DataSetVectorLabelC & dataSet, const IndexC & fv1, const IndexC & fv2)
  {

    // have to find min max
    RealRangeC xrange(RavlConstN::maxReal, -RavlConstN::maxReal);
    RealRangeC yrange(RavlConstN::maxReal, -RavlConstN::maxReal);
    for (DataSet2IterC<SampleVectorC, SampleLabelC> it(dataSet); it; it++) {
      xrange.Min() = Min(xrange.Min(), it.Data1()[0]);
      xrange.Max() = Max(xrange.Max(), it.Data1()[0]);
      yrange.Min() = Min(yrange.Min(), it.Data1()[1]);
      yrange.Max() = Max(yrange.Max(), it.Data1()[1]);
    }
    SetXRange(xrange);
    SetYRange(yrange);

    SArray1dC<FieldInfoC> fieldInfo = dataSet.Sample1().FieldInfo();

    FilenameC tmpFile = "/tmp/data";
    tmpFile = tmpFile.MkTemp();
    OStreamC os(tmpFile);
    StringC cmd;

    UIntT label = 0;
    UIntT count = 0;
    StringC bigCmd = "plot ";
    for (SArray1dIterC<SampleVectorC> it(dataSet.SeperateLabels()); it; it++) {
      StringC className = "Label " + (StringC)label;
      if(fieldInfo.IsValid()) {
        className = fieldInfo[label].Name();
      }
      StringC cmd;
      cmd.form("'%s' every ::%d::%d with points pointtype 1 title \'%s\'",
          tmpFile.data(),
          count,
          count + it.Data().Size() - 1,
          className.data());
      for (SampleIterC<VectorC> vecIt(it.Data()); vecIt; vecIt++) {
        os << vecIt.Data()[fv1] << ' ' << vecIt.Data()[fv2] << endl;
      }
      label++;
      count += it.Data().Size();
      bigCmd += cmd + ",";
    }
    bigCmd.del((int) bigCmd.Size().V() - 1, (int) 1);
    os.Close();
    Command(bigCmd);
    Flush();
    return true;
  }

  /*
   * Set the X label
   */
  bool GnuPlot2dC::SetXLabel(const StringC & xlabel)
  {
    Command("set xlabel \'" + xlabel + "\'");
    return false;
  }

  /*
   * Set the Y label
   */
  bool GnuPlot2dC::SetYLabel(const StringC & ylabel)
  {
    Command("set ylabel \'" + ylabel + "\'");
    return false;
  }

  /*
   *  Set axis range of a plot
   */

  bool GnuPlot2dC::SetXRange(const RealRangeC & xrange)
  {
    StringC cmd;
    cmd.form("set xrange [%f:%f]", xrange.Min(), xrange.Max());
    return Command(cmd);
  }

  /*
   *  Set axis range of a plot
   */

  bool GnuPlot2dC::SetYRange(const RealRangeC & yrange)
  {
    StringC cmd;
    cmd.form("set yrange [%f:%f]", yrange.Min(), yrange.Max());
    return Command(cmd);
  }

  /*
   * Set the line style
   */

  bool GnuPlot2dC::SetLineStyle(const StringC & lineStyle)
  {
    StringC cmd;
    cmd.form("set style data %s", lineStyle.data());
    return Command(cmd);
  }

  /*
   * Send the plotting device a generic command
   */

  bool GnuPlot2dC::Command(const StringC & command)
  {
    RavlInfo("gnuplot: '%s'", command.data());
    m_gnuPlot.StdIn() << command << endl;
    return false;
  }

  void GnuPlot2dC::Flush()
  {
    m_gnuPlot.StdIn().os().flush();
  }

} /* namespace RavlN */
