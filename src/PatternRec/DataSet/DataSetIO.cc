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

#define DODEBUG 0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlN {

  //! A function that saves a Vector dataset as a CSV file.
  bool SaveDataSetVectorLabel(const StringC & filename, const DataSetVectorLabelC & dataset) {

    // If we have csv extension lets save it as a csv file
    FilenameC fname(filename);
    if (fname.HasExtension("csv")) {
      return SaveDataSetVectorLabelCSV(filename, dataset);
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
  bool LoadDataSetVectorLabel(const StringC & filename, DataSetVectorLabelC & dataset) {

    // If we have csv extension lets save it as a csv file
    FilenameC fname(filename);
    if (fname.HasExtension("csv")) {
      return LoadDataSetVectorLabelCSV(filename, dataset);
    }

    IStreamC is(filename);
    if (!is.good()) {
      SysLog(SYSLOG_ERR, "Error opening file to load dataset '%s'", filename.data());
      return false;
    }
    is >> dataset;
    return true;
  }

  //! A function that saves a Vector dataset as a CSV file.
  bool SaveDataSetVectorLabelCSV(const StringC & filename, const DataSetVectorLabelC & dataset) {

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
    for (DataSet2IterC<SampleVectorC, SampleLabelC> it(dataset); it; it++) {
      for (Array1dIterC<RealT> vecIt(it.Data1()); vecIt; vecIt++) {
        os << *vecIt << ",";
      }
      // FIXME: We could get in trouble if DataT is not a scalar or string of some sort!
      os << it.Data2() << "\n";
    }

    return true;
  }

  //! A function that loads a DataSetVectorLabel from a CSV file.
  bool LoadDataSetVectorLabelCSV(const StringC & filename, DataSetVectorLabelC & dataset) {

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
      if (startLine == 2) {
        if (it.IsLast()) {
          labelFieldInfo = FieldInfoC(*it);
        } else {
          fieldInfo[c] = FieldInfoC(*it);
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
      for (DLIterC<StringC> it(line); it; it++) {
        if (c == vec.Size()) {
          label = it.Data().UIntValue();
        } else {
          vec[c] = it.Data().RealValue();
        }
        c++;
      }
      dataset.Append(vec, label);
    }
    return true;
  }

}
