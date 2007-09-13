#if !defined(LEGACYGRABFILEREADER_HH)
#define LEGACYGRABFILEREADER_HH
//! rcsid="$Id: $"
//! author="Simon Tredwell"
//! docentry="Ravl.API.Images.Video.Video IO.DVS"
//! lib=DVS

#include "Ravl/Image/GrabfileReader.hh"

#include <fstream>

#include "Ravl/Image/Utilities.hh"

namespace RavlImageN {

//class DVSBufferC;
//class CardModeC;

//! userlevel=Normal
//: Read legacy grabfiles generated by the DVS system.
// This class will read grabfiles generated prior to the new grabfile
// format being introduced.
class LegacyGrabfileReader: public GrabfileReader {
public:
  LegacyGrabfileReader()
    :
    GrabfileReader()
    //m_file(0)
  {
    // Do nothing
  }

  virtual ~LegacyGrabfileReader();

  //==========================================================================//

  //virtual bool Open(const char* const filename, CardModeC& mode);
  virtual bool Open(const char* const filename);
  //: Open file and read file header.

  virtual void Close();
  //: Close file.

  virtual bool Ok() const;
  //: Are there any problems with the IO?

  virtual bool HaveMoreFrames();

  //virtual bool GetNextFrame(DVSBufferC &buffer);
  virtual bool GetNextFrametest(BufferC<char> &bu, UIntT &vsize, UIntT &asize);
  //: Read the next frame to a buffer.

  virtual int Version() const {return 0;}
  //: The version of the reader.

  virtual ByteFormatT getByteFormat() { 
  cout << "legacy reader ByteFormat is " << byteformat << endl;
  return byteformat;}

  virtual ColourModeT getColourMode() { 
  cout << "legacy reader ColourMode is " << colourmode << endl;
  return colourmode;}

  //--------------------------------------------------------------------------//

protected:

  std::ifstream m_infile;

  //FILE * m_file;
  //: The file handle to be read from.
  int m_video_buffer_size;

  int m_audio_buffer_size;

  VideoModeT videomode;
  ByteFormatT byteformat;
  ColourModeT colourmode;

};

}


#endif // LEGACYGRABFILEREADER_HH