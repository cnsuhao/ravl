// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2002, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
////////////////////////////////////////////////////////////////////////////
//! author="Kieron J Messer"
//! lib=RavlDV
//! date="23/9/100"
//! rcsid="$Id$"

#include "Ravl/Image/DvDevice.hh"
#include "Ravl/Matrix.hh"

#include <unistd.h>
#include "Ravl/Assert.hh"
#include <netinet/in.h> 
#include <string.h> 
#include <time.h> 
#include <stdlib.h>

namespace RavlImageN {

int raw_iso_handler(raw1394handle_t handle, int channel, size_t length, quadlet_t *data)
{
  PalFrameC *frame = (PalFrameC*)raw1394_get_userdata(handle); //: get user data
  
  if (length > 16) {
    
    unsigned char *p = (unsigned char*) & data[3];
    int section_type = p[0] >> 5;   /* section type is in bits 5 - 7 */
    int dif_sequence = p[1] >> 4;   /* dif sequence number is in bits 4 - 7 */
    int dif_block = p[2];
    
    frame->writeDifBlock(section_type, dif_sequence, dif_block, length, p);
  }
  
  return 0;
}



void mysleep(int microseconds) {

  usleep((unsigned long)microseconds);

  //timespec delay; 
  //delay.tv_sec=(long int)floor(microseconds/1000);
  //delay.tv_nsec = (long int) (microseconds % 1000)*1e6;
  //nanosleep(&delay, NULL);   
}



DvDeviceC::DvDeviceC()
{
  
  device = -1;
  int i;
  
  // Declare some default values.
  
#ifdef RAW1394_V_0_8
  handle = raw1394_get_handle();
#else
  handle = raw1394_new_handle();
#endif
  if (!handle)
    {
      if (!errno)
        {
	  fprintf(stderr, "Not Compatable!\n");
        } else {
	  perror("Couldn't get 1394 handle");
	  fprintf(stderr, "Is ieee1394, driver, and raw1394 loaded?\n");
        }
      exit(1);
    } 
  
  if (raw1394_set_port(handle, 0) < 0) {
    perror("couldn't set port");
    raw1394_destroy_handle(handle);
    exit(1);
  }
  
  
  for (i=0; i < raw1394_get_nodecount(handle); ++i)
    {
      if (rom1394_get_directory(handle, i, &rom_dir) < 0)
    	{
	  fprintf(stderr,"error reading config rom directory for node %d\n", i);
	  raw1394_destroy_handle(handle);
	  exit(1);
        }
      
      if ( (rom1394_get_node_type(&rom_dir) == ROM1394_NODE_TYPE_AVC) &&
	   avc1394_check_subunit_type(handle, i, AVC1394_SUBUNIT_TYPE_VCR))
        {
	  device = i;
	  break;
        }
    }
  
  if (device == -1)
    {
      fprintf(stderr, "Could not find any AV/C devices on the 1394 bus.\n");
      raw1394_destroy_handle(handle);
      exit(1);
    }

  // lets set the ISO receiver
  raw1394_set_iso_handler(handle, 63, raw_iso_handler);
  raw1394_set_userdata(handle, (PalFrameC*)&frame);
}


//: OK a few check state routines
bool
DvDeviceC::isPlaying() const
{
  if(avc1394_vcr_is_playing(handle, device)) return true;
  return false;
}

bool
DvDeviceC::isRecording() const
{
  if(avc1394_vcr_is_recording(handle, device)) return true;
  return false;
}


void
DvDeviceC::Play() const
{
  avc1394_vcr_play(handle, device);
  return;
}

void
DvDeviceC::NextFrame() const
{
  avc1394_vcr_next(handle, device);
  return;
}


void
DvDeviceC::TrickPlay(const int speed) const
{
  //cout << "setting speed: " << speed << endl;
  avc1394_vcr_trick_play(handle, device, speed);
  return;
}


void
DvDeviceC::Pause() const
{
  avc1394_vcr_pause(handle, device);
  return;
}



void
DvDeviceC::Stop() const
{
  avc1394_vcr_stop(handle, device);
  return;
}

void
DvDeviceC::Rewind() const
{
  avc1394_vcr_rewind(handle, device);
  return;
}

void
DvDeviceC::ForwardWind() const
{
  avc1394_vcr_forward(handle, device);
  return;
}

       
TimeCodeC
DvDeviceC::getTimeCode() const
{
  char *tc =  avc1394_vcr_get_timecode(handle, device);
  TimeCodeC myTc(tc);
  return myTc;
}


void sleeping(int n) {
  for(IntT i=1;i<n;i++) {
    // this is daft
    MatrixC mat(100,100);
    mat.Fill(10);
    mat = mat * mat;
  }
}


void
DvDeviceC::gotoTimeCode(const TimeCodeC &tcTarget) const
{
  
 bool bForward = false;
 bool done = false;
 

  StringC status = Status();
  StringC want("Playing Paused");
  if(status!=want) {
    Pause();
    sleeping(200); // lets sleep
  }
  
  

  // Get the current timecode
  TimeCodeC tcCurrent = getTimeCode();
  cout << tcCurrent << endl;

  // Calculate how many frames to go
  int iTotNumOfFrames = tcCurrent.NumberOfFramesTo(tcTarget);
  if(tcCurrent < tcTarget) bForward = true;
  
  //: we could be there
  if(iTotNumOfFrames==0) {
    goto TCEND;
  }
    
  if(iTotNumOfFrames<125) goto FEW_FRAMES;
  else if(iTotNumOfFrames<150) goto VERY_NEAR;
  else if(iTotNumOfFrames<500) goto NEAR;
  // MORE THAN 20s AWAY
  
  // Do normal fast foward etc if more than 2 minutes away
  if(iTotNumOfFrames>=500) {
    
    if(bForward) TrickPlay(12);
    else TrickPlay(-12);
    
    while(!done) {
      TimeCodeC tcNow = getTimeCode();
      int iNumFrames = tcNow.NumberOfFramesTo(tcTarget);
      if(iNumFrames < 500) done = true;
      else mysleep(50);
    }
    iTotNumOfFrames=499;
  }

  
 NEAR:
  // LESS THAN 20s AWAY WE CAN SLOW DONE A LITTLE
  if((iTotNumOfFrames < 500) && (iTotNumOfFrames>=150)) {
    if(bForward) TrickPlay(10);
    else TrickPlay(-10);
    
    done = false;
    
    while(!done) {
      TimeCodeC tcNow = getTimeCode();
      int iNumFrames = tcNow.NumberOfFramesTo(tcTarget);
      //cout << iNumFrames << endl;
      if(iNumFrames < 150) done = true;
      else mysleep(50);
    }
    
    iTotNumOfFrames = 149;
  }

 VERY_NEAR:  
  // NOW LESS THAN 6s
  if((iTotNumOfFrames < 150) && (iTotNumOfFrames >= 50)) {
    if(bForward) TrickPlay(8);
    else TrickPlay(-8);
    
    done = false;
    
    while(!done) {
      TimeCodeC tcNow = getTimeCode();
      int iNumFrames = tcNow.NumberOfFramesTo(tcTarget);
      //cout << iNumFrames << endl;
      if(iNumFrames < 25) done = true;
      else mysleep(50);
    }
  }
 FEW_FRAMES:  
  // NOW IN FINAL SECOND
  if(bForward) TrickPlay(4);
  else TrickPlay(-4);
  
  done = false;
  
  while(!done) {
    TimeCodeC tcNow = getTimeCode();
    int iNumFrames = tcNow.NumberOfFramesTo(tcTarget);
    if(iNumFrames==0) {
      Pause();
      done = true;
    }
    else mysleep(50);
  }	
  
 TCEND:
  mysleep(50);
  
    
}


StringC
DvDeviceC::Status() const 
{
  quadlet_t  quad =avc1394_vcr_status(handle, device);
  StringC status(avc1394_vcr_decode_status(quad)); 
  return status;
}


TimeCodeC
DvDeviceC::grabFrame() 
{
  int channel=63;

  //: start ISO receive
  if (raw1394_start_iso_rcv(handle, channel) < 0) {
    cerr << "raw1394 - couldn't start iso receive" << endl;
    exit( -1);
  }

  bool done=false;
  TimeCodeC ret;
  while(!done) {    
    raw1394_loop_iterate(handle);
    if(frame.isValid()) {
      done=true;
      ret= frame.extractTimeCode();
      done = true;
    }
  }
  raw1394_stop_iso_rcv(handle, channel);
  
  return ret;
}


ImageC<ByteRGBValueC>
DvDeviceC::grabImage() 
{
  int channel=63;

  //: start ISO receive
  if (raw1394_start_iso_rcv(handle, channel) < 0) {
    cerr << "raw1394 - couldn't start iso receive" << endl;
    exit( -1);
  }

  bool done=false;
  while(!done) {    
    raw1394_loop_iterate(handle);
    if(frame.isValid()) {
      //TimeCodeC tcFrame = frame.extractTimeCode();
      //cout << "grabbed frame: " << tcFrame << endl;
      done = true;
    }
  }
  raw1394_stop_iso_rcv(handle, channel);

  return frame.Image();
}

ImageC<ByteRGBValueC>
DvDeviceC::grabFrame(const TimeCodeC & tcGrab) 
{
  int channel=63;
  gotoTimeCode(tcGrab);

  //: start ISO receive
  if (raw1394_start_iso_rcv(handle, channel) < 0) {
    cerr << "raw1394 - couldn't start iso receive" << endl;
    exit( -1);
  }

  bool done=false;
  while(!done) {    
    raw1394_loop_iterate(handle);
    if(frame.isValid()) {
      // TimeCodeC tcFrame = frame.extractTimeCode();
      //cout << "grabbed frame: " << tcFrame << endl;
      done = true;
    }
  }
  raw1394_stop_iso_rcv(handle, channel);

  return frame.Image();
}


bool
DvDeviceC::grabSequence(const char * filename, const TimeCodeC & tcStart, const TimeCodeC & tcEnd) 
{
  //: goto the first timecode
  TimeCodeC zero(0,0,0,0);
  TimeCodeC offset(0,0,3,0); // three seconds
  TimeCodeC realStart = tcStart-offset;
  if(realStart<zero)realStart=zero;
  bool dropped = false;
  //: first goto timecode
  gotoTimeCode(realStart);
  //: open the file for the video data
  FILE *fp = fopen(filename, "wb");
  int channel = 63;

  
  //: start ISO receive
  if (raw1394_start_iso_rcv(handle, channel) < 0) {
    cerr << "raw1394 - couldn't start iso receive" << endl;
    exit( -1);
  }
  
  int frames=0;
  bool done=false;
  //: OK lets go for it
  Play(); // lets start play
  TimeCodeC tcNext(tcStart);
  bool capture=false;
  while(!done) {    
    raw1394_loop_iterate(handle);
    
    if(frame.isValid()) {
      TimeCodeC tcFrame = frame.extractTimeCode();
      
      if(!capture) if(tcFrame>=tcStart) capture=true; //: look to start capturing
      
      if(capture) {
	if(tcFrame<tcNext) {
	  // cerr << "double frame..not saving" << endl;
	} else {
	  if(tcFrame>tcNext) {
	    cerr << "dropped frame" << tcNext << endl;
	    dropped = true;
	  } 
	  
	  fwrite(frame.getData(), 144000, sizeof(ByteT), fp);
	  frames++;
	  //frame.Decode();
	  if(tcFrame >= tcEnd) done=true;
	}
      }
      tcNext = tcFrame;
      tcNext+=1;
    }
    
  }
  
  fclose(fp);
  raw1394_stop_iso_rcv(handle, channel);
  //  cout << "grabbed frames: " << frames << endl;
  
  return dropped;
  
}

bool
DvDeviceC::grabWav(const char * filename, const TimeCodeC & tcStart, const TimeCodeC & tcEnd) 
{
  //: goto the first timecode
  TimeCodeC zero(0,0,0,0);
  TimeCodeC offset(0,0,3,0); // three seconds
  TimeCodeC realStart = tcStart-offset;
  if(realStart<zero)realStart=zero;
  bool dropped = false;
  //: first goto timecode
  gotoTimeCode(realStart);
  //: open the file for the video data
  int channel = 63;
  WavFileC audio;

  //: start ISO receive
  if (raw1394_start_iso_rcv(handle, channel) < 0) {
    cerr << "raw1394 - couldn't start iso receive" << endl;
    exit( -1);
  }
  

  int frames=0;
  bool done=false;
  //: OK lets go for it
  Play(); // lets start play
  TimeCodeC tcNext(tcStart);
  bool capture=false;
  bool firstframe=true;
  while(!done) {    
    raw1394_loop_iterate(handle);
    
    if(frame.isValid()) {
      TimeCodeC tcFrame = frame.extractTimeCode();
      
      if(firstframe) {
	// lets set audio properties
	UIntT bitsPerSample = frame.Decoder()->audio->quantization;
	UIntT sampleFrequency = frame.Decoder()->audio->frequency;
	UIntT numChannels =  frame.Decoder()->audio->num_channels;
	cout << sampleFrequency << " Hz" << endl;
	cout << bitsPerSample << " bits" << endl;
	cout << numChannels << " channels" << endl;
	FilenameC fn(filename);
	audio = WavFileC(fn, bitsPerSample, sampleFrequency, numChannels); 
	firstframe=false;
      }

      if(!capture) if(tcFrame>=tcStart) capture=true; //: look to start capturing
      
      if(capture) {

	if(tcFrame<tcNext) {
	  // cerr << "double frame..not saving" << endl;
	} else {
	  if(tcFrame>tcNext) {
	    cerr << "dropped frame" << tcNext << endl;
	    dropped = true;
	  } 
	  
	  audio.write(frame.Sound());
	  frames++;
	  //frame.Decode();
	  if(tcFrame >= tcEnd) done=true;
	}
      }
      tcNext = tcFrame;
      tcNext+=1;
    }
    
  }
  
  audio.Close();
  raw1394_stop_iso_rcv(handle, channel);
  //  cout << "grabbed frames: " << frames << endl;
  
  return dropped;
  
}

} // end namespace
 
