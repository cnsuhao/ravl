// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlAudioUtil

#include "Ravl/Audio/AudioFrame.hh"
#include "Ravl/BinStream.hh"

namespace RavlAudioN {

  //: Construct from components.
  //: --------------------------------------------------------------------------------------------------------------------------  
  AudioFrameBodyC::AudioFrameBodyC(const SArray1dC<ByteT> &data,
			   IntT nchannels,
			   RealT nfreq,
			   IntT nbits)
    : audio(data),
      channels(nchannels),
      freq(nfreq),
      bits(nbits)
  {}
  
  //: Constructor from stereo
  //: --------------------------------------------------------------------------------------------------------------------------
  AudioFrameBodyC::AudioFrameBodyC(const SArray1dC<SampleElemC<2,Int16T> > &data,RealT nfreq) 
    : stereoData(data),
      channels(2),
      freq(nfreq),
      bits(16)
  {}


  //: Constructor from stream
  //: --------------------------------------------------------------------------------------------------------------------------
  AudioFrameBodyC::AudioFrameBodyC ( BinIStreamC & stream ) 
{
  stream >> channels >> freq >> bits ; 
  if ( channels  == 2 ) 
    stream >> stereoData ; 
  else 
    stream >> audio ; 
}


  //: Constructor from binary stream 
  //: --------------------------------------------------------------------------------------------------------------------------
  AudioFrameBodyC::AudioFrameBodyC ( istream & stream ) 
{
 stream >> channels >> freq >> bits ; 
  if ( channels  == 2 ) 
    stream >> stereoData ; 
  else 
    stream >> audio ; 
}


  //: Save to stream 
  //: --------------------------------------------------------------------------------------------------------------------------
 bool AudioFrameBodyC::Save(ostream & stream) const
{
  stream << channels << " " << freq << " " << bits ; 
  if ( channels == 2 ) 
    stream << stereoData ; 
  else 
    stream << audio ; 
return true ; 
}


  //: Save to binary stream 
  //: --------------------------------------------------------------------------------------------------------------------------
  bool  AudioFrameBodyC::Save ( BinOStreamC & stream ) const 
{
  stream << channels << freq << bits ; 
  if ( channels == 2 ) 
    stream << stereoData ; 
  else 
    stream << audio ; 
return true ; 
}


}
