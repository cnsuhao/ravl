// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2002, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_SAMPLEIO_HEADERS
#define RAVL_SAMPLEIO_HEADERS 1
//! rcsid="$Id$"
//! author="Charles Galambos"
//! docentry="Ravl.API.Pattern Recognition.Data Set"
//! lib=RavlPatternRec
//! file="Ravl/PatternRec/DataSet/SampleIO.hh"

#include "Ravl/PatternRec/Sample.hh"
#include "Ravl/PatternRec/SampleIter.hh"
#include "Ravl/DP/SequenceIO.hh"
#include "Ravl/DArray1dIter.hh"
#include "Ravl/Vector.hh"
#include "Ravl/DLIter.hh"
#include "Ravl/OS/SysLog.hh"

namespace RavlN {

  template<class DataT>
  bool LoadSample(const StringC &filename, SampleC<DataT> &sample, UIntT maxSamples = ((UIntT) -1),
                  const StringC &format = "", bool verbose = false) {
    DPIPortC<DataT> ip;
    if (!OpenISequence(ip, filename, format, verbose))
      return false;
    while (maxSamples-- > 0) {
      DataT dat;
      if (!ip.Get(dat))
        break;
      sample.Append(dat);
    }
    return true;
  }
  //: Load sample from a file sequence 

  template<class DataT>
  bool SaveSample(const StringC &filename, const SampleC<DataT> &sample, const StringC &format = "", bool verbose =
                      false) {
    DPOPortC<DataT> ip;
    if (!OpenOSequence(ip, filename, format, verbose))
      return false;
    for (DArray1dIterC<DataT> it(sample.DArray()); it; it++) {
      if (!ip.Put(*it))
        return false;
    }
    return true;
  }

  //: Save sample to a CSV file

  bool SaveSampleCSV(const StringC & filename, const SampleC<VectorC> & sample, DListC<StringC> & headings) {

    OStreamC os(filename);
    if (!os.good())
      return false;

    // No point if empty sample size
    if (sample.Size() < 1) {
      SysLog(SYSLOG_ERR, "No samples to save to file '%s'", filename.data());
      return false;
    }

    // Put in some headings if required
    if (!headings.IsEmpty()) {

      // check dimensions match
      if (sample.First().Size() != headings.Size()) {
        SysLog(SYSLOG_ERR, "Samples dimension does not match number of headings");
        return false;
      }

      // save the headings
      for (DLIterC<StringC> it(headings); it; it++) {
        os << *it;
        if (!it.IsLast()) {
          os << ",";
        }
      }
      // and a new line
      os << "\n";
    }

    // Save the sample stream
    for (SampleIterC<VectorC> it(sample); it; it++) {
      for (Array1dIterC<RealT> vecIt(*it); vecIt; vecIt++) {
        os << *vecIt;
        if (!vecIt.IsLast()) {
          os << ",";
        }
      }
      os << "\n";
    }

    // All OK
    return true;
  }

}

#endif
