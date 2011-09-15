#include "Ravl/Option.hh"
#include "Ravl/XMLFactory.hh"
#include "Ravl/Resource.hh"
#include "Ravl/PatternRec/DataSetVectorLabel.hh"
#include "Ravl/PatternRec/DataSetIO.hh"
#include "Ravl/IO.hh"
#include "Ravl/OS/Filename.hh"

using namespace RavlN;

// Do some things with data sets

bool MakeTrainTestDataSet(DataSetVectorLabelC & dset, const FilenameC & outputFile) {

  SysLog(SYSLOG_INFO, "Making a training and test data set.");

  // What we will do is make two data sets - with equal samples per class
  UIntT minSamplesInAClass = dset.ClassNums()[dset.ClassNums().IndexOfMin()];
  SysLog(SYSLOG_INFO, "There are going to be %d samples per class", minSamplesInAClass/2);

  // Make sure we shuffle dataset first
  dset.Shuffle();
  dset = dset.ExtractPerLabel(minSamplesInAClass);

  // FIXME: Why doesn't inherit proper?
  DataSet2C<SampleVectorC, SampleLabelC> train = dset.ExtractSample(0.50);

  StringC path = outputFile.PathComponent();
  if (!path.IsEmpty()) {
    path = outputFile.PathComponent();
  }
  path += outputFile.BaseNameComponent();
  StringC trainFile = path + "_train." + outputFile.Extension();
  StringC testFile = path + "_test." + outputFile.Extension();
  SysLog(SYSLOG_INFO, "Saving training data set to '%s'", trainFile.data());
  SaveDataSetVectorLabel(trainFile, DataSetVectorLabelC(train.Sample1(), train.Sample2()));
  SysLog(SYSLOG_INFO, "Saving test data set to '%s'", testFile.data());
  SaveDataSetVectorLabel(testFile, dset);

  return true;

}

int main(int nargs, char **argv) {

  OptionC opts(nargs, argv);

  StringC installDir = opts.String("i", PROJECT_OUT, "Install directory.");
  RavlN::SetResourceRoot(installDir);
  DListC<StringC> sampleSets = opts.List("samples", "A list of sample sets to append into a master dataset.");
  DListC<StringC> datasets = opts.List("datasets", "A list of datasets to append into a master dataset.");
  DListC<StringC> dataFields = opts.List("dataFields", "Insert these fields into the final dataset");
  StringC labelField = opts.String("labelField","","Insert this as the field for the label");
  FilenameC outputFile = opts.String("o", "out.csv", "The output dataset!  ");
  bool equaliseSamples = opts.Boolean("eq", false, "Make sure we have an equal number of samples per class");
  bool makeTrainTest = opts.Boolean("tt", false, "Make a training and test data set");
  UIntT samplesPerClass = opts.Int("n", 0, "The number of samples per class");
  opts.Check();

  SysLogOpen("doDataSet", true, true, false, -1, false);

  try {
    SysLog(SYSLOG_INFO, "Ravl dataset manipulation program");

    DataSetVectorLabelC dset(10000);

    // Have we been asked to load in sample vectors
    if (opts.IsOnCommandLine("samples")) {
      UIntT label = 0;
      for (DLIterC<StringC> it(sampleSets); it; it++) {
        SysLog(SYSLOG_INFO, "Loading sample set '%s'", it.Data().data());
        SampleVectorC sample;
        if (!LoadSampleVector(*it, sample)) {
          SysLog(SYSLOG_ERR, "Trouble load sample set '%s'", it.Data().data());
          continue;
        }
        dset.Append(sample, label);
        label++;
      }
    }

    // Maybe we have been asked to load in datasets
    if (opts.IsOnCommandLine("datasets")) {
      for (DLIterC<StringC> it(datasets); it; it++) {
        SysLog(SYSLOG_INFO, "Loading dataset '%s'", it.Data().data());
        DataSetVectorLabelC localDataset;
        LoadDataSetVectorLabel(*it, localDataset);
        dset.Append(localDataset);
        dset.Sample1().SetFieldInfo(localDataset.Sample1().FieldInfo());
        dset.Sample2().SetFieldInfo(localDataset.Sample2().FieldInfo());
      }
    }

    // Check we have loaded some data from somewhere
    if (dset.Size() < 1) {
      SysLog(SYSLOG_ERR, "No samples in dataset.  Nothing to process!");
    }

    if (makeTrainTest) {
      MakeTrainTestDataSet(dset, outputFile);
      return 1;
    }

    if (equaliseSamples) {
      dset = dset.ExtractPerLabel(dset.ClassNums()[dset.ClassNums().IndexOfMax()]);
    }

    if (samplesPerClass > 0 && samplesPerClass < dset.ClassNums()[dset.ClassNums().IndexOfMin()]) {
      dset = dset.ExtractPerLabel(samplesPerClass);
    }

    // Have we been asked to attach some fields to the dataset?
    if(opts.IsOnCommandLine("dataFields")) {
      SArray1dC<FieldInfoC>fieldInfo(dataFields.Size());
      UIntT c=0;
      for(DLIterC<StringC>it(dataFields);it;it++) {
        fieldInfo[c] = FieldInfoC(*it);
        c++;
      }
      if(!dset.Sample1().SetFieldInfo(fieldInfo)) {
        SysLog(SYSLOG_ERR, "Trouble attaching fields to dataset.");
      }
    }

    // Set the label field
    if(opts.IsOnCommandLine("labelField")) {
      dset.Sample2().SetFieldInfo(FieldInfoC(labelField));
    }


    // And save the datasets
    SysLog(SYSLOG_INFO, "Saving data set '%s'", outputFile.data());
    SaveDataSetVectorLabel((StringC) outputFile, dset);

  } catch (const RavlN::ExceptionC &exc) {
    SysLog(SYSLOG_ERR, "Exception: %s", exc.Text());
  } catch (...) {
    SysLog(SYSLOG_ERR, "Unknown exception");
  }
}
