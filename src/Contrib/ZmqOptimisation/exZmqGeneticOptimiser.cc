#include "Ravl/Zmq/ZmqGeneticOptimiser.hh"
#include "Ravl/Genetic/GeneticOptimiserCheckPoint.hh"
#include "Ravl/Genetic/GenomeConst.hh"
#include "Ravl/Genetic/GenomeClass.hh"
#include "Ravl/Genetic/EvaluateFitnessFunc.hh"
#include "Ravl/Genetic/GeneFactory.hh"
#include "Ravl/RLog.hh"
#include "Ravl/Option.hh"
#include "Ravl/XMLFactory.hh"
#include "Ravl/Resource.hh"
#include "Ravl/DP/PrintIOInfo.hh"
#include "Ravl/Point2d.hh"
#include "Ravl/DP/TypeConverter.hh"
#include "Ravl/DP/Blackboard.hh"

#define CATCH_EXCEPTIONS 1

using namespace RavlN;
using namespace RavlN::GeneticN;
using namespace RavlN::ZmqN;

// Define how to construct our target object.
static GeneticN::GeneTypeFloatC::RefT g_numType = new GeneticN::GeneTypeFloatC("coord", -100.0, 100.0);

BlackboardC ConvertGeneFactory2Blackboard(const GeneticN::GeneFactoryC & factory)
{
  float x, y;
  factory.Get("x", x, *g_numType);
  factory.Get("y", y, *g_numType);
  //RavlInfo("Values: %f %f ",x,y);
  BlackboardC bb(true);
  bb.Put("row", (RealT)x);
  bb.Put("col", (RealT)y);
  return bb;
}
DP_REGISTER_CONVERSION(ConvertGeneFactory2Blackboard, 1.0);

int main(int nargs, char **argv)
{

  OptionC opt(nargs, argv);
  SetResourceRoot(opt.String("i", PROJECT_OUT, "Install location. "));
  StringC configFile = opt.String("c", Resource("Ravl/Zmq", "exZmqGeneticOptimiser.xml"), "Configuration file");
  StringC logFile = opt.String("l", "stderr", "Checkpoint log file. ");
  StringC logLevel = opt.String("ll", "info", "Logging level (debug, info, warning, error)");
  bool verbose = opt.Boolean("v", false, "Verbose mode.");
  opt.Check();

  RLogInit(nargs, argv, logFile.chars(), verbose);
  RLogSubscribeL(logLevel.chars());

#if CATCH_EXCEPTIONS
  try {
#endif

    XMLFactoryContextC factory(configFile);

    ZmqGeneticOptimiserC::RefT optimiser;

    if (!factory.UseComponent("ZmqOptimiser", optimiser)) {
      RavlError("Failed to find optimiser.");
      return 1;
    }

    // FIXME: Get CheckPoint working with derived classes!
#if 0
    GeneticOptimiserCheckPointC::RefT optimiserCheckPoint;
    if(!factory.UseComponent("OptimiserCheckPoint",optimiserCheckPoint)) {
      RavlError("Failed to find optimiser checkpoint.");
      return 1;
    }
#endif

    RavlInfo("Running optimisation.");

    optimiser->Run();

    RavlInfo("Optimisation complete");

#if CATCH_EXCEPTIONS
  } catch (...) {
    RavlError("Caught exception running model.");
  }
#endif

  return 0;
}

