// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU
// General Public License (GPL). See the gpl.licence file for details or
// see http://www.gnu.org/copyleft/gpl.html
// file-header-ends-here
#ifndef RAVL_DVDREAD_HEADER
#define RAVL_DVDREAD_HEADER 1
//! rcsid = "$Id$"
//! lib = RavlDVDRead
//! author = "Warren Moore"

#include "Ravl/DP/SPort.hh"
#include "dvdread/dvd_reader.h"

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

  protected:
    void Close();
    //: Close the DVD read objects
    
    bool ReadBlock(const UIntT block);
    //: Read a block into the cache
    
  protected:
    StringC m_device;                 // DVD device name
    UIntT m_title;                    // DVD title index
    dvd_reader_t *m_dvd;              // DVD read object
    dvd_file_t *m_file;               // DVD file object
    UIntT m_sizeBlocks;               // File size in blocks
    UIntT m_currentBlock;             // Current cached block
    UIntT m_currentByte;              // Current byte pos
    SArray1dC<ByteT> m_bufBlock;      // VOB block buffer
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

    DVDReadC(const UIntT title = 1, const StringC device = "/dev/dvd") :
      DPEntityC(*new DVDReadBodyC(title, device))
    {}
    //: Constructor.
    //!param: title The title track to read (default = 1)
    //!param: device A string naming the DVD device (default = /dev/dvd)
    
  };
  
}

#endif
