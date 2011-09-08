#include "Ravl/Option.hh"
#include "Ravl/XMLFactory.hh"
#include "Ravl/Resource.hh"
#include "Ravl/PatternRec/DesignClassifierSupervised.hh"
#include "Ravl/PatternRec/DataSetVectorLabel.hh"
#include "Ravl/PatternRec/Error.hh"
#include "Ravl/PatternRec/DataSet2Iter.hh"
#include "Ravl/Sums1d2.hh"
#include "Ravl/OS/SysLog.hh"

using namespace RavlN;

// Simple program which performs leave one out

int main(int nargs, char **argv) {

  OptionC opts(nargs, argv);

  StringC installDir = opts.String("i", PROJECT_OUT, "Install directory.");
  RavlN::SetResourceRoot(installDir);
  StringC configFile = opts.String("c", RavlN::Resource("Ravl/PatternRec", "classifier.xml"),
      "Classifier config file.  Look here for setting-up classifier parameters.");
  StringC classifierType = opts.String("classifier", "KNN", "The type of classifier to use [KNN|GMM|SVM].");
  StringC dsetFile = opts.String("dset", "", "The dataset to perform leave one out on!");
  //bool verbose = opts.Boolean("v", false, "Verbose mode.");
  UIntT maxIter = opts.Int("maxIter", 0, "Set the maximum number of iterations (0 do all)");
  opts.Check();

  try {
    XMLFactoryC::RefT mainFactory = new XMLFactoryC(configFile);
    XMLFactoryContextC context(*mainFactory);

    // Get classifier designer
    SysLog(SYSLOG_INFO,"Initialising classifier '%s'", classifierType.data());
    DesignClassifierSupervisedC design;
    if (!context.UseComponent(classifierType, design, true)) {
      SysLog(SYSLOG_ERR,"No '%s' component in XML config", classifierType.data());
      return 1;
    }

    // Get dataset
    SysLog(SYSLOG_INFO,"Loading dataset from file '%s'", dsetFile.data());
    DataSetVectorLabelC dset;
    IStreamC is(dsetFile);
    if (!is.good()) {
      SysLog(SYSLOG_ERR,"Trouble loading dataset from file!");
      return 1;
    }
    is >> dset;

    // Lets compute mean and variance of dataset and normalise input
    SysLog(SYSLOG_INFO,"Normalising sample!");
    MeanCovarianceC meanCovariance = dset.Sample1().MeanCovariance();
    dset.Sample1().Normalise(meanCovariance);

    UIntT outPos = 0;
    Sums1d2C sum;
    if (maxIter == 0 || maxIter > dset.Size()) {
      maxIter = dset.Size();
      SysLog(SYSLOG_INFO,"Using maximum number of samples '%d' in leave-one-out tests", maxIter);
    } else {
      // We need to shuffle the dataset so we do no get bias if only doing a sub-set
      dset.Shuffle();
      SysLog(SYSLOG_INFO,"Only using a sub-set of samples '%d' in leave one out test", maxIter);
    }

    SysLog(SYSLOG_INFO,"Performing leave-one-out test");
    for (DataSet2IterC<SampleVectorC, SampleLabelC> outIt(dset); outIt; outIt++) {

      // check if we have reached max iterations set by user
      if (outPos == maxIter)
        break;

      DataSetVectorLabelC trainDataSet(dset.Size() - 1);
      DataSetVectorLabelC testDataSet(1);
      cerr << "\rProcessing (" << outPos << "/" << maxIter << ")...." << (RealT) outPos / (RealT) dset.Size() * 100.0
          << "\%";
      // Build the dataset
      UIntT inPos = 0;
      for (DataSet2IterC<SampleVectorC, SampleLabelC> inIt(dset); inIt; inIt++) {
        // we have out test vector
        if (inPos == outPos) {
          testDataSet.Append(inIt.Data1(), inIt.Data2());
        } else {
          trainDataSet.Append(inIt.Data1(), inIt.Data2());
        }
        inPos++;
      }

      // Train classifier
      ClassifierC classifier = design.Apply(trainDataSet.Sample1(), trainDataSet.Sample2());

      // Lets get error
      ErrorC error;
      RealT pmc = error.Error(classifier, testDataSet);
      sum += pmc;
      outPos++;

    }
    cerr << endl;
    SysLog(SYSLOG_INFO,"The probability of miss-classification is %0.4f(%0.4f)", sum.MeanVariance().Mean(),
        sum.MeanVariance().Variance());

  } catch (const RavlN::ExceptionC &exc) {
    SysLog(SYSLOG_ERR,"Exception:%s", exc.Text());
  } catch (...) {
    SysLog(SYSLOG_ERR,"Unknown exception");
  }
}
