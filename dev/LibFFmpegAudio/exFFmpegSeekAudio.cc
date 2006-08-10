// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU
// General Public License (GPL). See the gpl.licence file for details or
// see http://www.gnu.org/copyleft/gpl.html
// file-header-ends-here
//////////////////////////////////////////////////////////////////
//! rcsid = "$Id: exFFmpegSeek.cc 4897 2005-04-23 14:30:05Z craftit $"
//! lib=RavlLibFFmpeg
//! author = "Warren Moore"
//! file="Ravl/Contrib/FFmpeg/exFFmpegSeek.cc"

#include "Ravl/Option.hh"
#include "Ravl/Image/LibFFmpegAudioFormat.hh"
#include "Ravl/IO.hh"
#include "Ravl/DP/SPort.hh"
#include "Ravl/OS/Date.hh"
#include "Ravl/OS/Filename.hh"
#include "Ravl/Random.hh"
#include "Ravl/DP/SequenceIO.hh"
#include "Ravl/Audio/Types.hh"

using namespace RavlN;
using namespace RavlImageN;
using namespace RavlAudioN;

int main(int nargs, char **argv)
{
  OptionC opts(nargs,argv);
  StringC filename = opts.String("", "in.avi", "Input AVI file.");
  opts.Check();
  
  // Check the file exists
  FilenameC fn(filename);
  if (!fn.Exists())
  {
    cerr << "Error opening file (" << filename << ")\n";
    return 1;
  }

  // Select the correct opening method
  FileFormatLibFFmpegAudioC format;
  DPIPortC< SArray1dC<SampleElemC<2,Int16T> > > in = format.CreateInput(filename,typeid(SArray1dC<SampleElemC<2,Int16T> >));
  if (!in.IsValid())
  {
    cerr << "Unable to open file (" << filename << ")\n";
    return 1;
  }
  DPISPortC< SArray1dC<SampleElemC<2,Int16T> > >  seek(in);
  if (!seek.IsValid())
  {
    cerr << "Unable to create seekable stream (" << filename << ")\n";
    return 1;
  }

  // Display the stream attributes
  DListC<StringC> attrs;
  if (in.GetAttrList(attrs))
  {
    DLIterC<StringC> it(attrs);
    while (it)
    {
      StringC value;
      if (in.GetAttr(*it, value))
        cerr << *it << " : " << value << endl;
      it++;
    }
  }
  
  // Number of frames to play with
  const IntT size = 1000;
  
  // Delay in seconds
  const RealT delay = 0.0;
  
  // Load the stream
  SArray1dC<SampleElemC<2,Int16T> > rgb;
  IntT count = 1000;
  
  DPOPortC<Int16T> out;
  if(!OpenOSequence(out,"/dev/dsp")) {
    cerr << "Failed to open output : " << "\n";
    return 1;}
  
  // Play forward, then back, then random for a bit
  while (true)
  {
    if (count > size * 3)
      count = 0;

    IntT frame = count;
    if (count >= size && count < size * 2)
      frame = size - (count - size);
    
    if (count >= size * 2)
      frame = (IntT)(Random1() * (RealT)size);
   
    count++;
    
    seek.Seek(frame);
      
    cerr << "==== Seeking to " << frame << endl;

    if(!seek.Get(rgb))
      break;
    for(SArray1dIterC<SampleElemC<2,Int16T> > it(rgb);it;it++){
      Sleep(1/48000);
  //    out.Put(it.Data().channel[0]);
    }
    
    //RavlN::Save("@X", rgb);

    Sleep(delay);
  }

  return 0;
}
