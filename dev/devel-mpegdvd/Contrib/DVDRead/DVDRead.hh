// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU
// General Public License (GPL). See the gpl.licence file for details or
// see http://www.gnu.org/copyleft/gpl.html
// file-header-ends-here
#ifndef RAVL_DVDREAD_HEADER
#define RAVL_DVDREAD_HEADER 1
//////////////////////////////////////////////////////////////////
//! rcsid = "$Id$"
//! lib = RavlDVDRead
//! author = "Warren Moore"

#include "Ravl/DP/SPort.hh"
#include "dvdread/dvd_reader.h"
#include "dvdread/ifo_types.h"
#include "Ravl/DArray1d.hh"
#include "Ravl/Tuple3.hh"

namespace RavlN
{
  
  class DVDReadBodyC :
    public DPISPortBodyC<ByteT>
  {
  public:
    DVDReadBodyC(const UIntT title, const StringC device);
    //: Constructor.
    
    ~DVDReadBodyC();
    //: Destructor.
    
    ByteT Get();
    //: Get a byte from the VOB data

    bool Get(ByteT &buff);
    //: Get a byte from the VOB data

    IntT GetArray(SArray1dC<ByteT> &data);
    //: Get an array of bytes from the VOB data

    bool Seek(UIntT off);
    //: Set the seek position

    UIntT Tell() const;
    //: Get the seek position

    UIntT Size() const;
    //: Get the complete size
    // Note: Currently this changes as more of the DVD is read

    bool Seek64(StreamPosT off);
    //: Set the seek position

    StreamPosT Tell64() const;
    //: Get the seek position

    StreamPosT Size64() const;
    //: Get the complete size
    
    bool IsGetEOS() const;
    //: Is it the EOS

    bool Discard();
    //: Discard the next data item
    
    bool GetAttr(const StringC &attrName, StringC &attrValue);
    //: Get an attribute
    
    StreamPosT SeekFrame(const StreamPosT frame);
    //: Move to the correct cell for the specified frame.
    //!return: Next frame actually that will be read

  protected:
    void Close();
    //: Close the DVD read objects
    
    Int64T GetTime(dvd_time_t time);
    //: Return the BCD coded time in seconds
    
    void BuildAttributes();
    //: Register stream attributes
    
  protected:
    StringC m_device;                                                     // DVD device name
    UIntT m_title;                                                        // DVD title index
    
    RealT m_numFps;                                                       // Title frames per second

    dvd_reader_t *m_dvdReader;                                            // DVD read object
    ifo_handle_t *m_dvdVmgFile;                                           // Video management info
    ifo_handle_t *m_dvdVtsFile;                                           // Video stream info
    pgc_t *m_dvdPgc;                                                      // Current PGC object
    dvd_file_t *m_dvdFile;                                                // DVD file object
    
    SArray1dC< Tuple3C< Int64T, StreamPosT, bool > > m_cellTable;
    // Table mapping playback times (in seconds) to sector positions with STC discontinuity indicator
    SArray1dC< Tuple3C< Int64T, StreamPosT, bool > > m_tmapTable;
    // Table mapping playback times (in seconds) to sector positions with STC discontinuity indicator
    
    StreamPosT m_curSector;                                               // Current sector number
    UIntT m_curByte;                                                      // Byte pointer in sector
  };

  class DVDReadC :
    public DPISPortC<ByteT>
  {
  public:
    DVDReadC() :
      DPEntityC(true)
    {}
    //: Default constructor.
    // Creates an invalid handle.

    DVDReadC(const UIntT title, const StringC device = "/dev/dvd") :
      DPEntityC(*new DVDReadBodyC(title, device))
    {}
    //: Constructor.
    //!param: title The title track to read (default = 1)
    //!param: device A string naming the DVD device (default = /dev/dvd)
    
    explicit DVDReadC(const DPISPortC<ByteT> &input) :
      DPEntityC(input)
    { if (dynamic_cast<DVDReadBodyC*>(&DPSeekCtrlC::Body()) == 0) Invalidate(); }
    //: Construct by upcasting from a pre-created DPISPort
    
  protected:
    DVDReadBodyC &Body()
    { return static_cast<DVDReadBodyC &>(DPIPortC<ByteT>::Body()); }
    //: Access body.

    const DVDReadBodyC &Body() const
    { return static_cast<const DVDReadBodyC &>(DPIPortC<ByteT>::Body()); }
    //: Access body.
  };
  
}

#endif
