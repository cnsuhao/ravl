// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2002, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_DATASETIO_HEADER
#define RAVL_DATASETIO_HEADER 1
//! rcsid="$Id$"
//! lib=RavlPatternRec
//! file="Ravl/PatternRec/DataSet/DataSetIO.hh"

#include "Ravl/PatternRec/DataSetVectorLabel.hh"
#include "Ravl/PatternRec/DataSet2Iter.hh"
#include "Ravl/PatternRec/SampleIO.hh"
#include "Ravl/Text/TextFile.hh"
#include "Ravl/OS/Filename.hh"
#include "Ravl/StringList.hh"

namespace RavlN {

  //! A function that saves a DataSetVectorLabel file.  If the filename has csv extension, it is saved in CSV format.
  bool SaveDataSetVectorLabel(const StringC & filename, const DataSetVectorLabelC & dataset);

  //! A function that loads a DataSetVectorLabel file.  If the filename has csv extension, it is loaded from CSV format.
  bool LoadDataSetVectorLabel(const StringC & filename, DataSetVectorLabelC & dataset);

  //! A function that saves a DataSetVectorLabel file as a CSV file.
  bool SaveDataSetVectorLabelCSV(const StringC & filename, const DataSetVectorLabelC & dataset,
                                 const DListC<StringC> & headings);

  //! A function that loads a DataSetVectorLabel from a CSV file.  It will try and automatically any headings on the
  //! the first line of the data file
  bool LoadDataSetVectorLabelCSV(const StringC & filename, DataSetVectorLabelC & dataset, DListC<StringC> & headings);

}

#endif
