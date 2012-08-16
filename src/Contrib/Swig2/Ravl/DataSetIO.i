// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2010, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html

%include "Ravl/Swig2/DataSetVectorLabel.i"

%{
#include "Ravl/PatternRec/DataSetIO.hh"
%}

namespace RavlN
{

   //! A function that saves a DataSetVectorLabel file.  If the filename has csv extension, it is saved in CSV format.
  bool SaveDataSetVectorLabel(const StringC & filename, const DataSetVectorLabelC & dataset);

  //! A function that loads a DataSetVectorLabel file.  If the filename has csv extension, it is loaded from CSV format.
  bool LoadDataSetVectorLabel(const StringC & filename, DataSetVectorLabelC & dataset);

  //! Load a DataSetVectorLabel and perform some common operations on it.
  bool LoadDataSetVectorLabel(const StringC & dataSetFile,
      bool shuffle,
      bool equaliseSamples,
      UIntT samplesPerClass,
      const SArray1dC<IndexC> & features,
      const FunctionC & normaliseFunc,
      DataSetVectorLabelC & dataSet);

  //! Load a DataSetVectorLabel. It will compute the chosen normalisation function on the loaded data.
  bool LoadDataSetVectorLabel(const StringC & dataSetFile,
      bool shuffle,
      bool equaliseSamples,
      UIntT samplesPerClass,
      const SArray1dC<IndexC> & features,
      DataSetNormaliseT normType,
      FunctionC & normaliseFunc,
      DataSetVectorLabelC & dataSet);

  //! A function that saves a DataSetVectorLabel file as a CSV file.
  bool SaveDataSetVectorLabelCSV(const StringC & filename, const DataSetVectorLabelC & dataset);

  //! A function that loads a DataSetVectorLabel from a CSV file.  It will try and automatically any headings on the
  //! the first line of the data file
  bool LoadDataSetVectorLabelCSV(const StringC & filename, DataSetVectorLabelC & dataset);

}
