// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2002, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlPatternRec
//! file="Ravl/PatternRec/DataSet/VectorLabelIO.cc"

#include "Ravl/PatternRec/DataSetIO.hh"
#include "Ravl/MeanCovariance.hh"
#include "Ravl/PatternRec/FuncLinear.hh"

#define DODEBUG 0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlN {

  bool LoadDataSetVectorLabelLIBSVM(const StringC & filename, DataSetVectorLabelC & dataset);

  //! A function that saves a Vector dataset as a CSV file.
  bool SaveDataSetVectorLabel(const StringC & filename, const DataSetVectorLabelC & dataset)
  {

    // If we have csv extension lets save it as a csv file
    FilenameC fname(filename);
    if (fname.HasExtension("csv")) {
      return SaveDataSetVectorLabelCSV(filename, dataset);
    }

    if (fname.HasExtension("abs")) {
      BinOStreamC os(filename);
      if (!os.Stream().good()) {
        SysLog(SYSLOG_ERR, "Error opening file to save dataset '%s'", filename.data());
        return false;
      }
      os << dataset;
    }

    OStreamC os(filename);
    if (!os.good()) {
      SysLog(SYSLOG_ERR, "Error opening file to save dataset '%s'", filename.data());
      return false;
    }
    os << dataset;
    return true;
  }

  //! A function that loads a DataSetVectorLabel file.
  bool LoadDataSetVectorLabel(const StringC & filename, DataSetVectorLabelC & dataset)
  {

    // If we have csv extension lets save it as a csv file
    FilenameC fname(filename);
    if (fname.HasExtension("csv")) {
      return LoadDataSetVectorLabelCSV(filename, dataset);
    }
    if (fname.HasExtension("libsvm"))
      return LoadDataSetVectorLabelLIBSVM(filename, dataset);

    IStreamC is(filename);
    if (!is.good()) {
      SysLog(SYSLOG_ERR, "Error opening file to load dataset '%s'", filename.data());
      return false;
    }
    is >> dataset;
    return true;
  }

  /*
   * Load a data set and get it ready
   */
  bool LoadDataSetVectorLabel(const StringC & dataSetFile,
      bool shuffle,
      bool equaliseSamples,
      UIntT samplesPerClass,
      const SArray1dC<IndexC> & features,
      const FunctionC & normaliseFunc,
      DataSetVectorLabelC & dataSet)
  {
    RavlInfo("Loading dataset from file '%s'", dataSetFile.data());

    // FIXME: Still want to use Load/Save instead
    if (!LoadDataSetVectorLabel(dataSetFile, dataSet)) {
      RavlError("Trouble loading data set from file '%s'", dataSetFile.data());
      return false;
    }

    // Shuffle
    if (shuffle) {
      RavlDebug("Shuffling data set.");
      dataSet.Shuffle();
    }

    // Equalise samples
    if (equaliseSamples) {
      UIntT min = dataSet.ClassNums()[dataSet.ClassNums().IndexOfMin()];
      RavlDebug( "Equalising number of samples per class to %d", min);
      dataSet = dataSet.ExtractPerLabel(min);
    }

    // Select number of samples per class
    if (samplesPerClass > 0 && samplesPerClass <= dataSet.ClassNums()[dataSet.ClassNums().IndexOfMin()]) {
      RavlDebug( "Setting the samples per class to %d", samplesPerClass);
      dataSet = dataSet.ExtractPerLabel(samplesPerClass);
    }

    // Select the features
    if (features.Size() > 0) {
      RavlInfo("Selecting '%s' features", StringOf(features.Size()).data());
      SampleVectorC vecs(dataSet.Sample1(), features);
      dataSet = DataSetVectorLabelC(vecs, dataSet.Sample2());
    }

    // Normalise
    if (normaliseFunc.IsValid()) {
      RavlDebug("Normalising data set....");
      dataSet.Sample1().Normalise(normaliseFunc);
    }

    return true;
  }

  /*
   * Load a data set and get it ready
   */
  bool LoadDataSetVectorLabel(const StringC & dataSetFile,
      bool shuffle,
      bool equaliseSamples,
      UIntT samplesPerClass,
      const SArray1dC<IndexC> & features,
      DataSetNormaliseT normType,
      FunctionC & normaliseFunc,
      DataSetVectorLabelC & dataSet)
  {

    /*
     * Load dataset and do no normalisation!
     */
    FunctionC function;
    if (!LoadDataSetVectorLabel(dataSetFile, shuffle, equaliseSamples, samplesPerClass, features, function, dataSet)) {
      return false;
    }

    /*
     * OK lets do some normalisation depending on what we have been asked
     */
    switch (normType)
    {
      case DATASET_NORMALISE_NONE:
        RavlInfo("Not doing any normalisation.  I hope you know what you are doing!");
        normaliseFunc = function;
        break;
      case DATASET_NORMALISE_MEAN: {
        RavlInfo("Normalising dataset using mean and covariance.");
        MeanCovarianceC meanCovariance = dataSet.Sample1().MeanCovariance();
        normaliseFunc = dataSet.Sample1().NormalisationFunction(meanCovariance);
        dataSet.Sample1().Normalise(meanCovariance);
      }
        break;
      case DATASET_NORMALISE_SCALE: {
        RavlInfo("Scaling the data set");
        FuncLinearC linearFunc;
        dataSet.Sample1().Scale(linearFunc);
        normaliseFunc = linearFunc;
      }
        break;
      default:
        // should never get here!
        RavlError("Unknown DataSet normalisation!");
    }

    return true;

  }

  //! A function that saves a Vector dataset as a CSV file.
  bool SaveDataSetVectorLabelCSV(const StringC & filename, const DataSetVectorLabelC & dataset)
  {

    // No point if empty sample size
    if (dataset.Size() < 1) {
      SysLog(SYSLOG_ERR, "No samples in dataset.  Nothing to save to file '%s'", filename.data());
      return false;
    }

    // Open output file ane check all OK!
    OStreamC os(filename);
    if (!os.good()) {
      SysLog(SYSLOG_ERR, "Error opening file to save dataset '%s'", filename.data());
      return false;
    }

    // If use has supplied headings, let put them in
    // Put in some headings if required
    SArray1dC<FieldInfoC> fieldInfo = dataset.Sample1().FieldInfo();
    if (fieldInfo.Size() != 0) {
      // check dimensions match
      if (dataset.Sample1().First().Size() != fieldInfo.Size()) {
        SysLog(SYSLOG_ERR, "Samples dimension does not match number of headings");
        return false;
      }

      // save the headings
      for (SArray1dIterC<FieldInfoC> it(fieldInfo); it; it++) {
        os << it.Data().Name();
        os << ",";
      }

      // and the field info for the label
      os << dataset.Sample2().FieldInfo().Name() << "\n";

    }

    // Lets iterate through data
    RCHashC<UIntT, StringC> label2names = dataset.Sample2().Label2ClassNames();
    for (DataSet2IterC<SampleVectorC, SampleLabelC> it(dataset); it; it++) {
      for (Array1dIterC<RealT> vecIt(it.Data1()); vecIt; vecIt++) {
        os << *vecIt << ",";
      }

      // Output label
      if (label2names.IsElm(it.Data2())) {
        os << it.Data2() << ":" << label2names[it.Data2()] << "\n";
      } else {
        os << it.Data2() << "\n";
      }

    }

    return true;
  }

  //! A function that loads a DataSetVectorLabel from a CSV file.
  bool LoadDataSetVectorLabelCSV(const StringC & filename, DataSetVectorLabelC & dataset)
  {

    // Open output file and check all OK!
    FilenameC fname(filename);
    if (!fname.Exists()) {
      SysLog(SYSLOG_ERR, "Data set file does not exist for loading '%s'", fname.data());
      return false;
    }

    TextFileC textFile(filename);
    // Now the first line might be the headings - we need to guess
    UIntT startLine = 1;
    StringListC headings(textFile[1].TopAndTail(), ",");
    for (DLIterC<StringC> it(headings); it; it++) {
      if (!it.Data().RealValue()) {
        startLine = 2;
        break;
      }
    }

    // Now lets put something into field information
    SArray1dC<FieldInfoC> fieldInfo(headings.Size() - 1);
    FieldInfoC labelFieldInfo;
    UIntT c = 0;
    for (DLIterC<StringC> it(headings); it; it++) {
      StringC featureLabel = it.Data().TopAndTail();
      if (startLine == 2) {
        if (it.IsLast()) {
          labelFieldInfo = FieldInfoC(featureLabel);
        } else {
          fieldInfo[c] = FieldInfoC(featureLabel);
        }
      } else {
        if (it.IsLast()) {
          labelFieldInfo = FieldInfoC("label");
        } else {
          fieldInfo[c] = "Unknown_" + (StringC) c;
        }
      }
      c++;
    }

    dataset = DataSetVectorLabelC(textFile.NoLines() - startLine + 1);
    dataset.Sample1().SetFieldInfo(fieldInfo);
    dataset.Sample2().SetFieldInfo(labelFieldInfo);
    for (UIntT i = startLine; i <= textFile.NoLines(); i++) {
      StringListC line(textFile[i], ",");
      VectorC vec(line.Size() - 1);
      UIntT c = 0;
      UIntT label;
      StringC className;
      for (DLIterC<StringC> it(line); it; it++) {
        if (c == vec.Size()) {
          if (it.Data().contains(":")) {
            StringListC ll(it.Data(), ":");
            label = ll.First().UIntValue();
            className = ll.Last().TopAndTail();
          } else {
            label = it.Data().UIntValue();
          }
        } else {
          vec[c] = it.Data().RealValue();
        }
        c++;
      }
      dataset.Append(vec, label);
      dataset.Sample2().SetClassName(label, className);
    } // end line
    return true;
  }

  //! A function that loads a DataSetVectorLabel from a CSV file.
  bool LoadDataSetVectorLabelLIBSVM(const StringC & filename, DataSetVectorLabelC & dataset)
  {
    RavlDebug("Reading %s in libsvm format. ", filename.c_str());
    IStreamC is(filename);
    if (!is.good()) {
      SysLog(SYSLOG_ERR, "Error opening file to load dataset '%s'", filename.data());
      return false;
    }
    size_t gVecSize = 0;
    size_t lineCount = 0;
    while (!is.eof()) {
      StringC str;
      if (readline(is, str, '\n', true) == 0)
        continue;
      //RavlDebug("Line '%s' ",str.c_str());
      StringListC list(str, " \t\n\r");
      DLIterC<StringC> it(list);
      for (it++; it; it++) {
        int ind = it->before(':').IntValue();
        if (ind == 0) {
          RavlError("Failed to find index in '%s' ", it->c_str());
          return false;
        }
        if (ind > (int) gVecSize)
          gVecSize = ind;
      }
      lineCount++;
    }
    dataset = DataSetVectorLabelC(lineCount);

    RavlDebug("Vector size:%d Lines:%u ", gVecSize, (unsigned) lineCount);
    is.Seek(0);
    is.is().clear();
    while (!is.eof()) {
      StringC str;
      if (readline(is, str, '\n', true) == 0)
        continue;
      str = str.TopAndTail();
      StringListC list(str, " \t\n\r");
      VectorC vec(gVecSize);
      vec.Fill(0);
      DLIterC<StringC> it(list);
      int label = it->IntValue();
      if (label < 0)
        label = 0;
      it++;
      for (; it; it++) {
        int ind = it->before(':').IntValue() - 1;
        if (ind < 0) {
          RavlError("Invalid index %d in '%s' from '%s' ", ind, it->c_str(), str.c_str());
          return false;
        }
        if (ind >= gVecSize) {
          RavlError("Index out of range %d  in '%s' from '%s' ", ind, it->c_str(), str.c_str());
          return false;
        }
        RealT val = it->after(':').RealValue();
        vec[ind] = val;
      }
      dataset.Append(vec, (unsigned) label);
    }
    return true;
  }

}
