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
#include "Ravl/Tuple2.hh"

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
  
    bool GetAttrList(DListC<StringC> &list) const;
    //: Get a list of attributes

  protected:
    void Close();
    //: Close the DVD read objects
    
    bool ReadCell(const UIntT cell);
    //: Read the info for a cell
    
  protected:
    StringC m_device;                             // DVD device name
    UIntT m_title;                                // DVD title index
    
    UIntT m_numChapters;                          // DVD chapter count for this title
    UIntT m_numAngles;                            // DVD angles for this title

    dvd_reader_t *m_dvdReader;                    // DVD read object
    ifo_handle_t *m_dvdVmgFile;                   // Video management info
    ifo_handle_t *m_dvdVtsFile;                   // Video stream info
    pgc_t *m_dvdCurPgc;                           // Current PGC object
    dvd_file_t *m_dvdFile;                        // DVD file object
    
    StreamPosT m_numCells;                        // Number of cells in title
    StreamPosT m_sizeCell;                        // Size (in blocks) of actual data in cell
    StreamPosT m_sizeData;                        // Size (in blocks) of actual data in title
    SArray1dC<StreamPosT> m_cellTable;            // Table mapping cell numbers to data sizes (in blocks)

    StreamPosT m_byteCurrent;                     // Byte currently sought to
    
    StreamPosT m_curCell;                         // Current cell info cached
    DArray1dC< Tuple2C<StreamPosT, StreamPosT> > m_navTable;
    // Table listing nav block locations (in absolute block offsets) and data sizes (in blocks) for the current cell
    StreamPosT m_curBlock;                        // Current block in cell cached
    SArray1dC<ByteT> m_curBlockBuf;               // Current block cache
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
