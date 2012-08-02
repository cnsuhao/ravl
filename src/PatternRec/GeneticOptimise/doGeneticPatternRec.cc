// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2011, Charles Galambos
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! lib=RavlGeneticOptimisation
//! author=Charles Galambos
//! docentry=Ravl.API.Math.Genetic.Optimisation


#include "Ravl/Genetic/GeneFactory.hh"
#include "Ravl/SysLog.hh"
#include "Ravl/Option.hh"
#include "Ravl/XMLFactory.hh"
#include "Ravl/Resource.hh"
#include "Ravl/DP/PrintIOInfo.hh"
#include "Ravl/Point2d.hh"
#include "Ravl/DP/TypeConverter.hh"
#include "Ravl/PatternRec/ClassifierPreprocess.hh"

#include "Ravl/PatternRec/GeneticOptimiseClassifierDesign.hh"
#include "Ravl/PatternRec/DataSetVectorLabel.hh"
#include "Ravl/PatternRec/DataSetIO.hh"
#include "Ravl/OS/Filename.hh"
#include "Ravl/IO.hh"

#define CATCH_EXCEPTIONS 1

using namespace RavlN::GeneticN;

int main(int nargs,char **argv)
{

  RavlN::OptionC opt(nargs,argv);
  RavlN::SetResourceRoot(opt.String("i", PROJECT_OUT, "Install location. "));
  RavlN::StringC configFile = opt.String("c", RavlN::Resource("Ravl/PatternRec", "doGeneticPatternRec.xml"), "Configuration file");
  bool listConv = opt.Boolean("lc",false,"List conversions");
  bool verbose = opt.Boolean("v",false,"Verbose logging.");
  RavlN::StringC trainingDataSetFile = opt.String("dset", "", "The dataset to train on.");
  RavlN::FilenameC classifierOutFile = opt.String("o", "classifier.abs", "Save classifier to this file.");

  opt.Check();

  RavlN::SysLogOpen("exGeneticOptimisation",false,true,true,-1,verbose);

  if(listConv) {
    RavlN::PrintIOConversions(std::cout);
    return 0;
  }
  RavlInfo("Started.");

#if CATCH_EXCEPTIONS
  try {
#endif

    RavlN::XMLFactoryContextC factory(configFile);

    RavlN::GeneticOptimiseClassifierDesignC designer;

    factory.UseComponent("Designer",designer);

    RavlInfo("Loading dataset from file '%s'", trainingDataSetFile.data());
    // FIXME: Still want to use Load/Save instead
    RavlN::DataSetVectorLabelC trainingDataSet;
    if (!RavlN::LoadDataSetVectorLabel(trainingDataSetFile, trainingDataSet)) {
      RavlError("Trouble loading dataset from file '%s'", trainingDataSetFile.data());
      return 1;
    }

    trainingDataSet.Shuffle(); // always good practice to shuffle (inplace)

    RavlN::MeanCovarianceC meanCovariance = trainingDataSet.Sample1().MeanCovariance();
    RavlN::FunctionC normaliseFunc = trainingDataSet.Sample1().NormalisationFunction(meanCovariance);
    trainingDataSet.Sample1().Normalise(meanCovariance);

    RavlN::ClassifierC bestClassifier;
    RavlN::RealT finalResult;

    if(!designer.Apply(trainingDataSet.Sample1(),
                   trainingDataSet.Sample2(),
                   bestClassifier,
                   finalResult)) {
      RavlError("Failed to design classifier.");
      return 1;
    }

    RavlN::ClassifierC classifier = RavlN::ClassifierPreprocessC(normaliseFunc, bestClassifier);

    // And save the classifier
    RavlInfo( "Saving classifier to '%s'", classifierOutFile.data());
    if (!RavlN::Save(classifierOutFile, classifier)) {
      RavlError( "Trouble saving classifier");
      return 1;
    }

#if CATCH_EXCEPTIONS
  } catch(...) {
    RavlError("Caught exception running model.");
  }
#endif

  return 0;
}

