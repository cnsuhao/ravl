// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU
// General Public License (GPL). See the gpl.licence file for details or
// see http://www.gnu.org/copyleft/gpl.html
// file-header-ends-here
//////////////////////////////////////////////////////////////////
//! rcsid = "$Id$"
//! lib = RavlDVDRead
//! author = "Warren Moore"

#include "Ravl/DVDRead.hh"
#include "dvdread/ifo_read.h"
#include "dvdread/ifo_print.h"

#define DODEBUG 0

#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlN
{
  
  DVDReadBodyC::DVDReadBodyC(const UIntT title, const StringC device) :
    m_device(device),
    m_title(title),
    m_dvd(NULL),
    m_file(NULL),
    m_sizeBlocks(0),
    m_currentBlock(0),
    m_currentByte(0)
  {
    // Try to open the DVD device
    m_dvd = DVDOpen(m_device);
    if (!m_dvd)
    {
      cerr << "Unable to open DVD device (" << m_device << ")" << endl;
      Close();
      return;
    }
    
    // Get the DVD info
    ifo_handle_t *vmg_file = ifoOpen(m_dvd, 0);
    if(!vmg_file)
    {
      cerr << "Can't open VMG info" << endl;
      Close();
      return;
    }
    tt_srpt_t *tt_srpt = vmg_file->tt_srpt;
  
    ONDEBUG(ifoPrint_TT_SRPT(tt_srpt);)
    ONDEBUG(cerr << "Found (" << tt_srpt->nr_of_srpts << ") titles on this DVD." << endl;)
    
    // Check we have valid title
    if (title < 1 || title > tt_srpt->nr_of_srpts)
    {
      cerr << "Invalid title number supplied (" << m_title << ")" << endl;
      Close();
      return;
    }
    
    ONDEBUG(cerr << "Title (" << m_title << ") has (" << tt_srpt->title[m_title].nr_of_ptts << ") chapters" << endl;)

    // Close the IFO
    ifoClose(vmg_file);
    
    // Open the title
    m_file = DVDOpenFile(m_dvd, m_title, DVD_READ_TITLE_VOBS);
    if (!m_file)
    {
      cerr << "Unable to open DVD title (" << m_title << ")" << endl;
      Close();
      return;
    }
    
    // Get the file size
    m_sizeBlocks = DVDFileSize(m_file);
  }
  
  DVDReadBodyC::~DVDReadBodyC()
  {
    Close();
  }
  
  void DVDReadBodyC::Close()
  {
    m_bufBlock = SArray1dC<ByteT>();
    m_title = 0;
    if (m_file)
    {
      DVDCloseFile(m_file);
      m_file = NULL;
    }
    if (m_dvd)
    {
      DVDClose(m_dvd);
      m_dvd = NULL;
    }
  }
  
  ByteT DVDReadBodyC::Get()
  {
    ByteT data = 0;
    Get(data);
    return data;
  }

  bool DVDReadBodyC::Get(ByteT &buff)
  {
    // Check we're not at the end of the stream
    if (IsGetEOS())
      return false;
    
    // Step to the block
    if (!ReadBlock(m_currentByte / DVD_VIDEO_LB_LEN))
      return false;
    
    // Get the byte
    buff = m_bufBlock[m_currentByte % DVD_VIDEO_LB_LEN];
    m_currentByte++;

    return true;
  }

  IntT DVDReadBodyC::GetArray(SArray1dC<ByteT> &data)
  {
    // Check we're not at the end of the stream
    if (IsGetEOS())
      return 0;
    
    // Make sure we have a buffer to read into
    if (!data.IsValid() || data.Size() == 0)
      return 0;
    
    // Step to the block
    UIntT dataRead = 0;
    while (dataRead < data.Size())
    {
      // Read the block
      if (!ReadBlock(m_currentByte / DVD_VIDEO_LB_LEN))
        break;
      
      // How much data do we need to copy from the block?
      UIntT size = DVD_VIDEO_LB_LEN - (m_currentByte % DVD_VIDEO_LB_LEN);
      if (data.Size() < size)
        size = data.Size();

      // Create the subarrays
      SizeBufferAccessC<ByteT> subDst = data.From(dataRead, size);
      SizeBufferAccessC<ByteT> subSrc = m_bufBlock.From(m_currentByte % DVD_VIDEO_LB_LEN, size);
      
      // Copy the data
      subDst.CopyFrom(subSrc);
      
      // Update the read positions
      m_currentByte += size;
      dataRead += size;
    }
    
    return dataRead;
  }

  bool DVDReadBodyC::Seek(UIntT off)
  {
    if (m_file)
    {
      m_currentByte = off;
      return true;
    }
    return false;
  }

  UIntT DVDReadBodyC::Tell() const
  {
    return m_currentByte;
  }

  UIntT DVDReadBodyC::Size() const
  {
    return m_sizeBlocks * DVD_VIDEO_LB_LEN;
  }

  //: Set the seek position
  
  bool DVDReadBodyC::Seek64(StreamPosT off) {
    if (m_file)
    {
      m_currentByte = off;
      return true;
    }
    return false;
  }
  
  //: Get the seek position
  
  StreamPosT DVDReadBodyC::Tell64() const {
    return m_currentByte;
  }
  
  //: Get the complete size
  
  StreamPosT DVDReadBodyC::Size64() const {
    return m_sizeBlocks * DVD_VIDEO_LB_LEN;    
  }
  
  
  bool DVDReadBodyC::ReadBlock(const UIntT block)
  {
    RavlAssertMsg(m_file, "DVDReadBodyC::ReadBlock requires open file");
    
    // Make sure we're not at the end
    if (block > m_sizeBlocks)
      return false;
    
    // Make sure we have a buffer object
    bool cached = true;
    if (!m_bufBlock.IsValid())
    {
      // Create the cache
      m_bufBlock = SArray1dC<ByteT>(DVD_VIDEO_LB_LEN);
      cached = false;
    }
    
    // Only read the block if it's not cached
    if (!cached || (m_currentBlock != block))
    {
      // Read the block
      DVDReadBlocks(m_file, block, 1, &(m_bufBlock[0]));
      m_currentBlock = block;
    }
    
    return true;
  }
  
  bool DVDReadBodyC::IsGetEOS() const
  {
    // EOS if no file open
    if (!m_file)
      return true;
    
    // EOS if at last block
    if (m_currentByte > (m_sizeBlocks * DVD_VIDEO_LB_LEN))
      return true;
    
    return false;
  }

  bool DVDReadBodyC::Discard()
  {
    // Check we're not at the end of the stream
    if (IsGetEOS())
      return false;
    
    m_currentByte++;
    
    return true;
  }
  
}
