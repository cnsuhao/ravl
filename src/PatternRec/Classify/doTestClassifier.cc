#include "Ravl/Option.hh"
#include "Ravl/XMLFactory.hh"
#include "Ravl/RLog.hh"
#include "Ravl/Resource.hh"
#include "Ravl/PatternRec/DesignClassifierSupervised.hh"
#include "Ravl/PatternRec/DataSetVectorLabel.hh"
#include "Ravl/PatternRec/ClassifierPreprocess.hh"
#include "Ravl/PatternRec/Error.hh"
#include "Ravl/PatternRec/DataSet2Iter.hh"
#include "Ravl/IO.hh"
#include "Ravl/OS/Filename.hh"

using namespace RavlN;

// Simple program which performs leave one out

int main(int nargs, char **argv) {

  OptionC opts(nargs, argv);

  StringC installDir = opts.String("i", PROJECT_OUT, "Install directory.");
  RavlN::SetResourceRoot(installDir);
  StringC classifierFile = opts.String("classifier", "", "Load the trained classifier from this file.");
  StringC dsetFile = opts.String("dset", "", "The dataset to perform the test on!");
  bool verbose = opts.Boolean("v", false, "Verbose mode.");
  StringC logFile = opts.String("l", "stderr", "Checkpoint log file. ");
  StringC logLevel = opts.String("ll", "debug", "Logging level (debug, info, warning, error)");

  opts.Check();

  RLogInit(nargs, argv, logFile.chars(), verbose);
  RLogSubscribeL(logLevel.chars());

  try {

    // And save the classifier
    ClassifierC classifier;
    rInfo("Loading classifier from '%s'", classifierFile.data());
    if (!Load(classifierFile, classifier)) {
      rError("Trouble loading classifier");
      return 1;
    }

    // Get dataset
    rInfo("Loading dataset from file '%s'", dsetFile.data());
    // FIXME: Want to use Load/Save instead
    DataSetVectorLabelC testDataSet;
    IStreamC is(dsetFile);
    if (!is.good()) {
      rError("Trouble loading dataset from file!");
      return 1;
    }
    is >> testDataSet;

    // Lets get error on the test data set
    ErrorC error;
    RealT pmc = error.Error(classifier, testDataSet);
    rInfo("The probability of miss-classification is %0.4f ", pmc);

  } catch (const RavlN::ExceptionC &exc) {
    rError("Exception:%s", exc.Text());
  } catch (...) {
    rError("Unknown exception");
  }
}
