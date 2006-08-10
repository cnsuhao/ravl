#include "Ravl/Option.hh"
#include "Ravl/Image/LibFFmpegAudioFormat.hh"
#include "Ravl/IO.hh"
#include "Ravl/DP/SPort.hh"
#include "Ravl/OS/Date.hh"
#include "Ravl/OS/Filename.hh"
#include "Ravl/Random.hh"
#include "Ravl/DP/SequenceIO.hh"
#include "Ravl/DP/AttributeType.hh"

#include "Ravl/GUI/Manager.hh"
#include "Ravl/GUI/Window.hh"
#include "Ravl/GUI/Slider.hh"
#include "Ravl/GUI/LBox.hh"
#include "Ravl/GUI/PackInfo.hh"

#include "Ravl/Audio/Types.hh"

using namespace RavlN;
using namespace RavlImageN;
using namespace RavlGUIN;
using namespace RavlAudioN;

RealT frameRate = 25.0;
DPISPortC< SArray1dC<SampleElemC<2,Int16T> > > in;

bool AudioSeek(RealT &frameNum){
  cerr << Round(in.Start()+frameNum) << "\n";
  in.Seek(Round(in.Start()+frameNum));
    return true;
}

int main(int nargs, char **argv)
{
  OptionC opts(nargs,argv);
  StringC filename = opts.String("", "in.avi", "Input AVI file.");
  bool seekStart = opts.Boolean("ss",false,"Seek to Start. ");
  opts.Check();
  
  // Check the file exists
  FilenameC fn(filename);
  if (!fn.Exists())
  {
    cerr << "Error opening file (" << filename << ")\n";
    return 1;
  }
  
  FileFormatLibFFmpegAudioC format;
  in = format.CreateInput(filename,typeid(SArray1dC<SampleElemC<2,Int16T> >));
  if (!in.IsValid())
  {
    cerr << "Unable to open file (" << filename << ")\n";
    return 1;
  }
  
  SArray1dC<SampleElemC<2,Int16T> > rgb;
  
  Manager.Init(nargs,argv); 
  WindowC win(400,400,"Audio Player"); 
  SliderC scale = SliderH(1,0.1,in.Size(),1,&AudioSeek);
  win.Add(VBox(PackInfoC(scale,false,true)));
  win.Show();
  
  cerr << in.Size();
  
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
  
  RealT sampleRate = 10000;
  in.GetAttr("samplerate", sampleRate);
  cerr << sampleRate;
  
  if(seekStart)
    in.Seek(in.Start()+1);
  
  DPOPortC<SampleElemC<2,Int16T> > out;
  if(!OpenOSequence(out,"@DEVAUDIO:/dev/dsp")) {
   cerr << "Failed to open output : " << "\n";
  return 1;}
  
  out.SetAttr("samplerate",sampleRate);
  Manager.Execute();
  Sleep(1);
  while(true){
    in.Get(rgb); 
  
  for(SArray1dIterC<SampleElemC<2,Int16T> > it(rgb);it;it++){
    out.Put(*it);
  }
  
  scale.UpdateValue(in.Tell()-in.Start());
  }
  Manager.Shutdown();
 
  return 0;
}
