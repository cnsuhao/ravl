#include "Ravl/Plot/GnuPlot3d.hh"
#include "Ravl/OS/Filename.hh"
#include "Ravl/RLog.hh"
#include "Ravl/HashIter.hh"
#include "Ravl/IO.hh"
#include "Ravl/OS/ChildOSProcess.hh"
#include "Ravl/OS/Date.hh"

namespace RavlGUIN {

  using namespace RavlN;
  using namespace RavlImageN;

  //! Constructor.

  GnuPlot3dC::GnuPlot3dC(const StringC & title, const StringC & xlabel, const StringC & ylabel, const StringC & zlabel) :
      m_terminal("x11 persist raise"), m_title(title), m_xlabel(xlabel), m_ylabel(ylabel), m_zlabel(zlabel)
  {

  }

  //! Make a plot
  bool GnuPlot3dC::AddPoint(const StringC & plotName, const Point3dC & point3d)
  {

    if (!IsElm(plotName)) {
      rInfo("Creating new plot %s", StringOf(plotName).data());
      Insert(plotName, DListC<Point3dC>());
    }
    (*this)[plotName].InsLast(point3d);
    return true;
  }

  //! Make a plot
  bool GnuPlot3dC::Plot(const StringC & outfile) const
  {

    rInfo("Attempting to perform 3d plot...");

    // Need to copy data to data files
    RCHashC<StringC, FilenameC> dataFiles;
    for (HashIterC<StringC, DListC<Point3dC> > it(*this); it; it++) {
      rInfo("Plotting '%s'", it.Key().data());
      FilenameC tmpData = "/tmp/" + it.Key();
      tmpData = tmpData.MkTemp(4);
      dataFiles.Insert(it.Key(), tmpData);
      OStreamC ofsData(tmpData);
      for (DLIterC<Point3dC> pt(it.Data()); pt.IsElm(); pt.Next())
        ofsData << *pt << endl;
    }

    FilenameC main("/tmp/gnu");
    main = main.MkTemp(4);
    rDebug("Main gnu file %s", main.data());
    {
      OStreamC ofsGnu(main);

      //: Print out the commands
      ofsGnu << "set title \'" << m_title << "\'" << endl;
      ofsGnu << "set xlabel \'" << m_xlabel << "\'" << endl;
      ofsGnu << "set ylabel \'" << m_ylabel << "\'" << endl;
      ofsGnu << "set zlabel \'" << m_zlabel << "\'" << endl;
      if (!outfile.IsEmpty())
        ofsGnu << "set output \'" << outfile << "\'" << endl;
      ofsGnu << "set terminal " << m_terminal << endl;
      ofsGnu << "set grid" << endl;
      ofsGnu << "set view 60,15" << endl;

      StringC with = "points";
      UIntT count = 0;
      ofsGnu << "splot ";
      for (HashIterC<StringC, FilenameC> it(dataFiles); it; it++) {

        if ((*this)[it.Key()].IsEmpty()) {
          rDebug("No data, no point in plotting!");
          continue;
        }

        // If it is not the first plot we need to add a comma
        if (count != 0) {
          ofsGnu << ", ";
        }

        // If we are plotting multiple graphs with points
        // then let change the shape for each plot
        StringC localWith = with;
        if (with == "points") {
          localWith = "points pointtype " + (StringC) (count + 1);
        }
        ofsGnu << "\'" << *it << "\' with " << localWith;
        // Put in a nice title for the legend
        ofsGnu << " title \"" << it.Key() << "\"";
        count++;
      }
    }

    //: Run gnuplot
    StringC com = "gnuplot " + main;
    rInfo("Plotting using command '%s'", com.data());
    system(com);
    Sleep(1);

    //: Then remove tmp files
    if (!main.Remove()) {
      rWarning("Trouble removing temp file %s", main.data());
    }

    for (HashIterC<StringC, FilenameC> it(dataFiles); it; it++) {
      if (!it.Data().Remove()) {
        rWarning("Trouble removing temp file %s", it.Data().data());
      }
    }

    return true;
  }

    // Get an image of the plot
    ImageC<ByteRGBValueC> GnuPlot3dC::Image(UIntT rows, UIntT cols)
    {

      // Sort out the terminal
      StringC term;
      term.form("png size %d, %d", cols, rows);
      m_terminal = term;

      // Make the temporary png file
      FilenameC pngFile = "/tmp/plot.png";
      pngFile = pngFile.MkTemp();

      // Do the plot
      ImageC<ByteRGBValueC> image;
      if (!Plot(pngFile)) {
        rError("Trouble performing plot");
        return image;
      }

      // Load image
      if (!RavlN::Load(pngFile, image)) {
        rError("Trouble loading image %s", pngFile.data());
        return image;
      }

      // Remove png file
      pngFile.Remove();

      // Reset terminal to x11
      m_terminal = "x11 persist raise";

      // and return
      return image;
    }


}

