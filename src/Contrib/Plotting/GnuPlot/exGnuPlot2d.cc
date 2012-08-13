// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, James Smith
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
/////////////////////////////////////////////////////////
//! rcsid="$Id$"
//! lib=RavlURLIO
//! file="Ravl/Contrib/Plotting/GnuPlot/exGnuPlot.cc"
//! author="Kieron Messer"
//! docentry="Ravl.Contrib.Plotting.GnuPlot"
//! userlevel=Normal

#include "Ravl/Plot/GnuPlot2d.hh"
#include "Ravl/Option.hh"
#include "Ravl/EntryPnt.hh"
#include "Ravl/StdConst.hh"
#include "Ravl/IO.hh"
#include "Ravl/SArray1d.hh"
#include "Ravl/Collection.hh"
#include "Ravl/SArray1dIter.hh"
#include "Ravl/Vector2d.hh"

using namespace RavlN;
using namespace RavlImageN;

int exGnuPlot(int nargs, char *args[])
{

  // Get command-line options
  OptionC opt(nargs, args);

  opt.Check();

  // Plot a Sine and Cosine on same graph
  SArray1dC<CollectionC<Point2dC> > points(3);
  for (SArray1dIterC<CollectionC<Point2dC> > it(points); it; it++) {
    it.Data() = CollectionC<Point2dC>(1000);
  }

  for (RealT ang = -4.0 * RavlConstN::pi; ang <= 4.0 * RavlConstN::pi; ang += RavlConstN::pi / 64.0) {
    points[0].Append(Point2dC(ang, Sin(ang)));
    points[1].Append(Point2dC(ang, Cos(ang)));
    points[2].Append(Point2dC(ang, Sin(2. * Sin(2. * Sin(2. * Sin(ang))))));
  }
#if 1
  GnuPlot2dC plot0("Sin(x)");
  plot0.SetLineStyle("points");

  //plot0.PlotFunction("sin(x)");
  //plot0.PlotFunction("cos(x)");
  plot0.SetXLabel("bananas");
  plot0.Plot(points[0].SArray1d(), "My Data");

  GnuPlot2dC plot1("Cos(x)");
  plot1.SetLineStyle("lines");
  plot1.Plot(points[1].SArray1d());

  /*
   * We can just plot a function
   */
  GnuPlot2dC plot2("My Function");
  plot2.PlotFunction("cos(x)/sin(x)");

  /*
   * We can just use the command function to do it all as well..
   */
  GnuPlot2dC plot3("A pretty function!");
  plot3.Command("set size ratio -1");
  plot3.Command("set nokey");
  plot3.Command("set noxtics");
  plot3.Command("set noytics");
  plot3.Command("set noborder");
  plot3.Command("set parametric");
  plot3.Command("x(t) = (R-r)*cos(t) + p*cos((R-r)*t/r)");
  plot3.Command("y(t) = (R-r)*sin(t) - p*sin((R-r)*t/r)");
  plot3.Command("gcd(x,y) = (x%y==0 ? y : gcd(y,x%y))");
  plot3.Command("R = 100; r = -49; p = -66; res = 10");
  plot3.Command("rr = abs(r)");
  plot3.Command("nturns = rr / gcd(R,rr)");
  plot3.Command("samp = 1 + res * nturns");
  plot3.Command("set samples samp");
  plot3.Command("plot [t=0:nturns*2*pi] x(t),y(t)");
#endif

  /*
   * What about a data set
   */
  DataSetVectorLabelC dset(1000);
  for (UIntT i = 0; i < 1000; i++) {
    dset.Append(Vector2dC(3.0 + RandomGauss(), 3.0 + RandomGauss()), 0);
    dset.Append(Vector2dC(RandomGauss(), RandomGauss()), 1);
  }
  GnuPlot2dC scatterPlot("A scatter plot");
  scatterPlot.ScatterPlot(dset);

  return 0;
}

RAVL_ENTRY_POINT(exGnuPlot);
