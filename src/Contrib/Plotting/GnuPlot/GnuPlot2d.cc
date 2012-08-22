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
#include "Ravl/OS/Filename.hh"

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
   * Plot as separate plots on same graph
   */
  bool GnuPlot2dC::Plot(const RCHashC<StringC, CollectionC<Point2dC> > & data)
  {

    // check it is still running
    if (!m_gnuPlot.IsRunning()) {
      RavlError("GnuPlot not running!");
      return false;
    }

    /*
     * Not a big fan of this but can not pipe data to GnuPlot
     * for more complicated plots...
     */
    FilenameC tmpFile = "/tmp/data";
    tmpFile = tmpFile.MkTemp(6, -1);
    OStreamC os(tmpFile);
    StringC cmd = "plot ";
    UIntT i = 0;
    for (HashIterC<StringC, CollectionC<Point2dC> > hshIt(data); hshIt; hshIt++) {
      os << "# " << hshIt.Key() << endl;
      StringC localPlot;
      localPlot.form("\'%s\' index %d title '%s',", tmpFile.data(), i, hshIt.Key().data());
      cmd += localPlot;
      for (SArray1dIterC<Point2dC> it(hshIt.Data().SArray1d()); it; it++) {
        StringC d;
        d.form("%f %f", it.Data().Row(), it.Data().Col());
        os << d << endl;
      }
      i++;
      os << "\n\n"; // double line important
    }
    cmd.del((int) cmd.Size() - 1, (int) 1);
    // send the command
    Command(cmd);

    Flush();
    return true;
  }

  /*
   * Plot a function
   */
  bool GnuPlot2dC::Plot(const StringC & function)
  {
    m_gnuPlot.StdIn() << "plot " << function << endl;
    Flush();
    return true;
  }

  /*
   * Plot a straight line
   */
  bool GnuPlot2dC::Plot(const LineABC2dC & line)
  {
    SArray1dC<Point2dC> arr(100);
    RealT step = m_xrange.Size() / 100.0;
    UIntT c = 0;
    for (RealT x = m_xrange.Min(); x <= m_xrange.Max(); x += step) {
      RealT y = line.ValueY(x);
      arr[c] = Point2dC(x, y);
      c++;
    }

    return Plot(arr);
  }

  /*
   * Make a scatter plot of a data set
   */
  bool GnuPlot2dC::Plot(const DataSetVectorLabelC & dataSet, UIntT fv1, UIntT fv2, UIntT samplesPerClass)
  {

    if (fv1 >= dataSet.Sample1().VectorSize() || fv2 >= dataSet.Sample1().VectorSize()) {
      RavlError("Feature index larger than in data set");
      return false;
    }

    DataSetVectorLabelC useDataSet = dataSet;
    if (samplesPerClass != 0) {
      useDataSet = dataSet.ExtractPerLabel(samplesPerClass);
    }

    SArray1dC<FieldInfoC> fieldInfo = dataSet.Sample1().FieldInfo();
    if (fieldInfo.IsValid()) {
      SetXLabel(fieldInfo[fv1].Name());
      SetYLabel(fieldInfo[fv2].Name());
    }

    // have to find min max
    RealRangeC xrange(RavlConstN::maxReal, -RavlConstN::maxReal);
    RealRangeC yrange(RavlConstN::maxReal, -RavlConstN::maxReal);
    for (DataSet2IterC<SampleVectorC, SampleLabelC> it(dataSet); it; it++) {
      xrange.Min() = Min(xrange.Min(), it.Data1()[fv1]);
      xrange.Max() = Max(xrange.Max(), it.Data1()[fv1]);
      yrange.Min() = Min(yrange.Min(), it.Data1()[fv2]);
      yrange.Max() = Max(yrange.Max(), it.Data1()[fv2]);
    }
    SetXRange(xrange);
    SetYRange(yrange);
    //: Plot a function

    FilenameC tmpFile = "/tmp/data";
    tmpFile = tmpFile.MkTemp(6, -1);
    OStreamC os(tmpFile);
    StringC cmd;

    UIntT label = 0;
    UIntT count = 0;
    StringC bigCmd = "plot ";
    for (SArray1dIterC<SampleVectorC> it(useDataSet.SeperateLabels()); it; it++) {
      StringC className;
      if (!dataSet.Sample2().GetClassName(label, className)) {
        className = "Label " + (StringC) label;
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
    m_xrange = xrange;
    StringC cmd;
    cmd.form("set xrange [%f:%f]", xrange.Min(), xrange.Max());
    return Command(cmd);
  }

  /*
   *  Set axis range of a plot
   */

  bool GnuPlot2dC::SetYRange(const RealRangeC & yrange)
  {
    m_yrange = yrange;
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
   * Set the output.  Either X11 or a filename.
   */

  bool GnuPlot2dC::SetOutput(const StringC & output, const IndexRange2dC & rec)
  {
    if (output == "x11" || output == "") {
      Command("set terminal wxt persist raise");
    } else {
      FilenameC fn(output);
      StringC cmd;
      if (fn.HasExtension("png")) {
        cmd.form("set terminal png size %d, %d", rec.Cols(), rec.Rows());
        Command(cmd);
      } else if (fn.HasExtension("jpg") || fn.HasExtension("jpeg")) {
        cmd.form("set terminal jpg size %d, %d", rec.Cols(), rec.Rows());
        Command(cmd);
      } else {
        RavlError("gnuplot terminal not supported yet '%s'.", output.data());
        return false;
      }
      cmd.form("set output \'%s\'", output.data());
      Command(cmd);
    }
    return true;
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
