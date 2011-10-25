// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2011, Charles Galambos
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! lib=RavlGeneticOptimisation
//! author=Charles Galambos
//! docentry=Ravl.API.Math.Genetic.Optimisation


#include "Ravl/Genetic/GeneticOptimiser.hh"
#include "Ravl/Genetic/GenomeConst.hh"
#include "Ravl/Genetic/GenomeClass.hh"
#include "Ravl/Genetic/EvaluateFitnessFunc.hh"
#include "Ravl/OS/SysLog.hh"
#include "Ravl/Option.hh"
#include "Ravl/XMLFactory.hh"
#include "Ravl/Resource.hh"
#include "Ravl/DP/PrintIOInfo.hh"
#include "Ravl/Point2d.hh"
#include "Ravl/OS/SysLog.hh"

#define CATCH_EXCEPTIONS 1

using namespace RavlN::GeneticN;


float EvaluateFitness(RavlN::Point2dC &point) {
  static RavlN::Point2dC target(0.1234,0.4321);
  return static_cast<float>(1.0/(point.EuclideanDistance(target) + 0.0001));
}

// Define how to construct our target object.
static RavlN::GeneticN::GeneTypeFloatC::RefT g_numType = new RavlN::GeneticN::GeneTypeFloatC("coord",0.0,1.0);
static RavlN::TypeNameC g_type1(typeid(RavlN::Point2dC),"RavlN::Point2dC");

RavlN::Point2dC ConvertGeneFactory2Point2d(const RavlN::GeneticN::GeneFactoryC &factory)
{
  float x,y;
  factory.Get("x",x,*g_numType);
  factory.Get("y",y,*g_numType);
  return RavlN::Point2dC(x,y);
}

DP_REGISTER_CONVERSION(ConvertGeneFactory2Point2d,1.0);


int main(int nargs,char **argv)
{
  RavlN::SysLogOpen("exGeneticOptimisation");

  RavlN::OptionC opt(nargs,argv);
  RavlN::SetResourceRoot(opt.String("i", PROJECT_OUT, "Install location. "));
  RavlN::StringC configFile = opt.String("c", RavlN::Resource("Ravl/Genetic", "exGeneticOptimisation.xml"), "Configuration file");
  bool listConv = opt.Boolean("lc",false,"List conversions");
  opt.Check();

  if(listConv) {
    RavlN::PrintIOConversions(std::cout);
    return 0;
  }
  RavlSysLogf(RavlN::SYSLOG_INFO,"Started.");

#if CATCH_EXCEPTIONS
  try {
#endif

    RavlN::XMLFactoryC::RefT mainFactory = new RavlN::XMLFactoryC(configFile);
    RavlN::XMLFactoryContextC factory(*mainFactory);

    GeneticOptimiserC::RefT optimiser;

    if(!factory.UseComponent("Optimiser",optimiser)) {
      RavlSysLogf(RavlN::SYSLOG_ERR,"Failed to find optimiser.");
      return 1;
    }

    optimiser->SetFitnessFunction(*new RavlN::GeneticN::EvaluateFitnessFuncC<RavlN::Point2dC>(&EvaluateFitness));

    RavlSysLogf(RavlN::SYSLOG_INFO,"Running optimisation.");

    optimiser->Run();

    RavlSysLogf(RavlN::SYSLOG_INFO,"Optimisation complete");

#if CATCH_EXCEPTIONS
  } catch(...) {
    RavlSysLogf(RavlN::SYSLOG_ERR,"Caught exception running model.");
  }
#endif

  return 0;
}

