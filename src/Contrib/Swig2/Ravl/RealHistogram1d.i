// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html


%include "Ravl/Swig2/SArray1d.i"
%include "Ravl/Swig2/DList.i"

%{
#ifdef SWIGPERL
#undef Copy
#endif

#include "Ravl/RealHistogram1d.hh"
#include "Ravl/SArray1dIter.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN {
	
  class RealHistogram1dC : public SArray1dC<RealT> {
  public:
    RealHistogram1dC();
    //: Default constructor.
    
    RealHistogram1dC(RealT min, RealT max, UIntT noOfBins);
   
    //RealHistogram1dC(RealT nscale, RealT noffset, const SArray1dC<UIntC> &array);
    //: Create a histogram from a bin width, an offset and an existing  SArray1dC of binned values.

    RealT Scale() const;
    //: Scaling - i.e. bin width.
    
    RealT Offset() const;
    //: Offset used in table.
    
    IndexC Bin(RealT v) const;
    //: Get the bin which value 'v' falls into.
    
    RealT MidBin(IndexC bin) const;
    //: Get the middle of given bin.
    
    RealT MinBin(IndexC bin) const;
    //: Get the lower limit of given bin.
    
    RealT MaxBin(IndexC bin) const;
    //: Get the upper limit of given bin.
    
    RealT MinLimit() const;
    //: Lower limit on values in the histogram range.

    RealT MaxLimit() const;
    //: Lower limit on values in the histogram range.
    
    void Reset();
    //: Reset counters in histogram to zero.
    
    void Vote(RealT v);
    //: Vote for value.
    // Note, this will not check that the value is within the histogram.
    // In check mode this will cause an error, in optimised it will corrupt
    // memory.
    
    void Vote(RealT v,IntT n);
    //: Vote for value n times.
    // Note, this will not check that the value is within the histogram.
    // In check mode this will cause an error, in optimised it will corrupt
    // memory.
    
    bool CheckVote(RealT v);
    //: Vote for value.
    // Returns false if value is out of range.
    
    bool CheckVote(RealT v,IntT n);
    //: Vote for value n times.
    // Returns false if value is out of range.

    //bool ArrayVote(const Array1dC<RealT> &data);
    //: Add to histogram bins using "data"
    // Returns false if any values from "data" are out of range.
    
    UIntT TotalVotes() const;
    //: Find the total number of votes cast.
    // This is computed not stored, and so is relatively slow.
    
    RealT Information() const;
    //: Calculate the amount of information represented by the histogram.
    // This is also known as the entropy of the histogram.
    
    RealT Energy() const;
    //: Calculate the energy represented by the original signal.

    //MeanVarianceC MeanVariance() const;
    //: Calculate the mean and variance for the signal.
    
    RealT SmoothedPDF(IntT bin,RealT sigma = 1) const;
    //: Evaluate histogram as a smoothed pdf.
    
    DListC<RealT> Peaks(UIntT width,UIntT threshold = 0) const;
    //: Find a list of peaks in the histogram.
    // The peaks are bigger than 'threshold' and larger than all those within +/- width.
    
    bool MinMax(IndexC &min,IndexC &max) const;
    //: Find the minimum and maximum bins with votes in.
    // Returns false if the histogram is empty, true otherwise.
    
    UIntT Sum(IndexC min,IndexC max) const;
    //: Sum votes in the bins from min to max inclusive.

  };
  
    
}
