#include "Ravl/Plot/GnuPlot3d.hh"
#include "Ravl/IO.hh"
#include "Ravl/RLog.hh"
#include "Ravl/Option.hh"
#include "Ravl/RandomGauss.hh"

using namespace RavlN;
using namespace RavlGUIN;
using namespace RavlImageN;

int main(int nargs, char **argv) {

  //: parse options
  OptionC opt(nargs, argv);
  StringC logFile = opt.String("l", "stderr", "Log file. ");
  StringC logLevel = opt.String("ll", "info", "Logging level. ");
  bool verbose = opt.Boolean("v", true, "Verbose, Output context information. ");
  opt.Check();

  RavlN::RLogInit(nargs, argv, logFile.chars(), verbose);
  RavlN::RLogSubscribeL(logLevel.chars());

  GnuPlot3dC plot("Big Test", "apples", "bananas", "pears");

  for (UIntT i = 0; i < 1000; i++) {
    plot.AddPoint(RandomGauss(), RandomGauss(), RandomGauss());
  }

  for (UIntT i = 0; i < 1000; i++) {
    plot.AddPoint("Plot 1", RandomGauss() + 10.0, RandomGauss() - 5.0, RandomGauss() + 13.0);
  }

  for (UIntT i = 0; i < 1000; i++) {
    plot.AddPoint("Plot 2", RandomGauss() - 10.0, RandomGauss() - 5.0, RandomGauss() - 10.0);
  }

  // Perform a plot. This should display plot in GnuPlot window
  plot.Plot();

  // Get plot as image
  ImageC<ByteRGBValueC> image = plot.Image(500, 500);
  Save("@X", image);

  return 0;
}
