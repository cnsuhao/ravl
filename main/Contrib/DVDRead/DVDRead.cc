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
#include "dvdread/nav_read.h"
#include "Ravl/DList.hh"

#define DODEBUG 0

#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlN
{
  
  const static UIntT g_startChapter = 1; // 1 <= chapter <= m_numChapters
  const static UIntT g_angle = 1;
  
  DVDReadBodyC::DVDReadBodyC(const UIntT title, const StringC device) :
    m_device(device),
    m_title(title - 1),
    m_numChapters(0),
    m_numAngles(0),
    m_dvdReader(NULL),
    m_dvdVmgFile(NULL),
    m_dvdVtsFile(NULL),
    m_dvdCurPgc(NULL),
    m_dvdFile(NULL),
    m_numCells(0),
    m_sizeCell(0),
    m_sizeData(0),
    m_byteCurrent(0),
    m_curCell(-1),
    m_curBlock(-1),
    m_curBlockBuf(DVD_VIDEO_LB_LEN)
  {
    // Try to open the DVD device
    m_dvdReader = DVDOpen(m_device);
    if (!m_dvdReader)
    {
      cerr << "Unable to open DVD device (" << m_device << ")" << endl;
      Close();
      return;
    }
    
    // Get the VMG IFO
    m_dvdVmgFile = ifoOpen(m_dvdReader, 0);
    if(!m_dvdVmgFile)
    {
      cerr << "DVDReadBodyC::DVDReadBodyC can't open VMG info" << endl;
      Close();
      return;
    }
    tt_srpt_t *tt_srpt = m_dvdVmgFile->tt_srpt;
    
    // Check the title is valid
    if (m_title > tt_srpt->nr_of_srpts)
    {
      cerr << "DVDReadBodyC::DVDReadBodyC invalid title set" << endl;
      Close();
      return;
    }
  
    // Get the title info
    m_numChapters = tt_srpt->title[m_title].nr_of_ptts;
    m_numAngles = tt_srpt->title[m_title].nr_of_angles;
    UIntT vtsnum = tt_srpt->title[m_title].title_set_nr;
    UIntT ttnnum = tt_srpt->title[m_title].vts_ttn;

    ONDEBUG(cerr << "DVDReadBodyC::DVDReadBodyC title(" << ttnnum << ") VTS(" << vtsnum << ") " << endl;)
    ONDEBUG(cerr << "DVDReadBodyC::DVDReadBodyC chapters(" << m_numChapters << ") angles(" << m_numAngles << ")" << endl;)

    // Check the angles
    if (m_numAngles == 0 || m_numAngles > 1)
    {
      cerr << "DVDReadBodyC::DVDReadBodyC zero/multiple angles not supported" << endl;
      Close();
      return;
    }
      
    // Get the VTS IFO
    m_dvdVtsFile = ifoOpen(m_dvdReader, vtsnum);
    if(!m_dvdVtsFile)
    {
      cerr << "DVDReadBodyC::DVDReadBodyC unable to load IFO for VTS " << vtsnum << endl;
      Close();
      return;
    }
    
    vts_ptt_srpt_t *vts_ptt_srpt = m_dvdVtsFile->vts_ptt_srpt;
    UIntT pgc_id = vts_ptt_srpt->title[ttnnum - 1].ptt[g_startChapter - 1].pgcn;
    m_dvdCurPgc = m_dvdVtsFile->vts_pgcit->pgci_srp[pgc_id - 1].pgc;
    m_numCells = m_dvdCurPgc->nr_of_cells;
    ONDEBUG(cerr << "DVDReadBodyC::DVDReadBodyC PGC(" << pgc_id << ") cells(" << m_numCells << ")" << endl;)
                
    // Open the title
    m_dvdFile = DVDOpenFile(m_dvdReader, vtsnum, DVD_READ_TITLE_VOBS);
    if (!m_dvdFile)
    {
      cerr << "DVDReadBodyC::DVDReadBodyC unable to open VTS(" << vtsnum << ") for title (" << ttnnum << ")" << endl;
      Close();
      return;
    }

    // Create the cell data table
    m_cellTable = SArray1dC<StreamPosT>(m_numCells);
    m_cellTable.Fill(-1);
    
    // Read in the first cell
    ReadCell(0);
    m_cellTable[0] = m_sizeCell;
    m_sizeData = m_sizeCell;
    
    /*
    // Read each cell info (but don't read/cache the data)
    for (UIntT i = 0; i < m_numCells; i++)
    {
      if (!ReadCell(i))
      {
        cerr << "DVDReadBodyC::DVDReadBodyC unable to read cell(" << i << ")" << endl;
        Close();
        return;
      }
      
      m_sizeData += m_sizeCell;
      m_cellTable[i] = m_sizeData;
    }
    ONDEBUG(cerr << "DVDReadBodyC::DVDReadBodyC total data(" << m_sizeData << ")" << endl;)
    */
  }
  
  DVDReadBodyC::~DVDReadBodyC()
  {
    Close();
  }
  
  void DVDReadBodyC::Close()
  {
    // Reset the params
    m_device = "";
    m_title = 0;
    m_numChapters = 0;
    m_numAngles = 0;
    m_numCells = 0;
    m_sizeCell = 0;
    m_sizeData = 0;
    m_byteCurrent = 0;
    m_curCell = -1;
    m_curBlock = -1;
    
    // Free all open handles
    if (m_dvdFile)
    {
      DVDCloseFile(m_dvdFile);
      m_dvdFile = NULL;
    }
    
    if (m_dvdVtsFile)
    {
      ifoClose(m_dvdVtsFile);
      m_dvdVtsFile = NULL;
    }
    
    if (m_dvdVmgFile)
    {
      ifoClose(m_dvdVmgFile);
      m_dvdVmgFile = NULL;
    }

    if (m_dvdReader)
    {
      DVDClose(m_dvdReader);
      m_dvdReader = NULL;
    }
  }
  
  ByteT DVDReadBodyC::Get()
  {
    SArray1dC<ByteT> data(1);
    GetArray(data);
    return data[0];
  }

  bool DVDReadBodyC::Get(ByteT &buff)
  {
    // Check we're not at the end of the stream
    if (IsGetEOS())
      return false;
    
    SArray1dC<ByteT> data(1);
    bool read = GetArray(data) > 0;
    if (read)
      buff = data[0];
    return read;
  }

  IntT DVDReadBodyC::GetArray(SArray1dC<ByteT> &data)
  {
    // Check we're not at the end of the stream
    if (IsGetEOS())
      return 0;
    
    // Which cell is the seek byte in
    StreamPosT dataRead = 0;
    while (dataRead < data.Size())
    {
      StreamPosT curCell = 0;
      StreamPosT curDataSize = 0;
      for (UIntT i = 0; i < m_numCells; i++)
      {
        // Check if the current cell has been read
        if (m_cellTable[i] == -1)
        {
          if (!ReadCell(i))
          {
            cerr << "DVDReadBodyC::GetArray unable to read cell(" << i << ")" << endl;
            return 0;
          }
          m_sizeData += m_sizeCell;
          m_cellTable[i] = m_sizeData;
        }
        
        // Stop if we've gone past the seek byte
        if (m_cellTable[i] * DVD_VIDEO_LB_LEN > m_byteCurrent)
          break;
        
        // Stop permanently if we've gone past the end
        if (i + 1 == m_numCells)
          break;
        
        // Store the required cell
        curCell++;
        curDataSize = m_cellTable[i];
      }
  
      // Read the current cell info
      if (curCell != m_curCell)
      {
        if (!ReadCell(curCell))
          return 0;
      }
      
      // Which block is the current byte in
      StreamPosT curNavBlock = 0;
      StreamPosT curNavSize = 0;
      for (UIntT i = 0; i < m_navTable.Size(); i++)
      {
        // Get the current offset
        curNavBlock = m_navTable[i].Data1();
        
        // Stop if we've gone past the seek byte
        if ((curDataSize + m_navTable[i].Data2()) * DVD_VIDEO_LB_LEN > m_byteCurrent)
          break;
        
        // Stop permanently if we've gone past the end
        if (i + 1 == m_navTable.Size())
        {
          cerr << "DVDReadBodyC::GetArray attempting to seek past end of cell" << endl;
          break;
        }
        
        // Store the required NAV block offset
        curNavSize = m_navTable[i].Data2();
        curDataSize += m_navTable[i].Data2();
      }
      
      // Identify the data block within the NAV block
      StreamPosT curBlock = (m_byteCurrent / DVD_VIDEO_LB_LEN) - curDataSize;
      ONDEBUG(cerr << "DVDReadBodyC::GetArray cell(" << m_curCell << ") nav block(" << curNavBlock << ") block(" << curBlock << ")" << endl;)
      
      // Cache the block
      if (m_curBlock != curBlock)
      {
        StreamPosT len = DVDReadBlocks(m_dvdFile, curNavBlock + curBlock + 1, 1, &(m_curBlockBuf[0]));
        if (len != 1)
        {
          cerr << "DVDReadBodyC::GetArray unable to read block data" << endl;
          Close();
          return 0;
        }
        m_curBlock = curBlock;
      }
      
      // Read out as much data as possible
      StreamPosT writeSize = DVD_VIDEO_LB_LEN - (m_byteCurrent % DVD_VIDEO_LB_LEN);
      if ((data.Size() - dataRead) < writeSize)
        writeSize = (data.Size() - dataRead);
  
      // Create the subarrays
      SizeBufferAccessC<ByteT> subDst = data.From(dataRead, writeSize);
      SizeBufferAccessC<ByteT> subSrc = m_curBlockBuf.From(m_byteCurrent % DVD_VIDEO_LB_LEN, writeSize);
        
      // Copy the data
      subDst.CopyFrom(subSrc);
        
      // Update the read positions
      m_byteCurrent += writeSize;
      dataRead += writeSize;
    }

    return dataRead;
  }

  bool DVDReadBodyC::Seek(UIntT off)
  {
    if (off < m_sizeData * DVD_VIDEO_LB_LEN)
    {
      m_byteCurrent = off;
      return true;
    }
    
    return false;
  }

  UIntT DVDReadBodyC::Tell() const
  {
    return m_byteCurrent;
  }

  UIntT DVDReadBodyC::Size() const
  {
    return m_sizeData * DVD_VIDEO_LB_LEN;
  }

  bool DVDReadBodyC::Seek64(StreamPosT off)
  {
    if (off < m_sizeData * DVD_VIDEO_LB_LEN)
    {
      m_byteCurrent = off;
      return true;
    }
    
    return false;
  }
  
  StreamPosT DVDReadBodyC::Tell64() const
  {
    return m_byteCurrent;
  }
  
  StreamPosT DVDReadBodyC::Size64() const
  {
    return m_sizeData * DVD_VIDEO_LB_LEN;
  }

  bool DVDReadBodyC::ReadCell(const UIntT cell)
  {
    RavlAssertMsg(m_dvdFile, "DVDReadBodyC::ReadCell requires open file");
    RavlAssertMsg(m_dvdCurPgc != NULL, "DVDReadBodyC::ReadCell requires valid current PGC");

    // Create the cell data store
    m_navTable = DArray1dC< Tuple2C<StreamPosT, StreamPosT> >(0);
    
    m_sizeCell = 0;
    StreamPosT pos = m_dvdCurPgc->cell_playback[cell].first_sector;
    while (true)
    {
      ByteT navData[DVD_VIDEO_LB_LEN];
      dsi_t dsiPack;
      StreamPosT nextVobu, nextIlvuStart, dataSize;

      // Read NAV packet
      StreamPosT len = DVDReadBlocks(m_dvdFile, (UIntT)pos, 1, navData);
      if(len != 1)
      {
        cerr << "DVDReadBodyC::ReadCell unable to read blocks" << endl;
        Close();
        return false;
      }

      // Parse the contained DSI packet
      navRead_DSI(&dsiPack, &(navData[DSI_START_BYTE]));
      RavlAssertMsg(pos == dsiPack.dsi_gi.nv_pck_lbn, "DVDReadBodyC::ReadCell DSI packet info does not match cell number");

      // Where do we go next
      nextIlvuStart = pos + dsiPack.sml_agli.data[g_angle].address;
      dataSize = dsiPack.dsi_gi.vobu_ea;
      m_sizeCell += dataSize;

      // Store the positions
      Tuple2C<StreamPosT, StreamPosT> info(pos, dataSize);
      m_navTable.Append(info);
      
      // Either step to the next data block or step past the end of the cell
      if(dsiPack.vobu_sri.next_vobu != SRI_END_OF_CELL)
        nextVobu = pos + (dsiPack.vobu_sri.next_vobu & 0x7fffffff);
      else
        break;

      // Skip to the next NAV block
      pos = nextVobu;
    }
    
    // Store the current cached cell number
    m_curCell = cell;
    
    ONDEBUG(cerr << "DVDReadBodyC::ReadCell cell(" << cell << ") start(" << m_dvdCurPgc->cell_playback[cell].first_sector << ") end(" << m_dvdCurPgc->cell_playback[cell].last_sector << ") data(" << m_sizeCell << ")" << endl;)

    return true;
  }
  
  bool DVDReadBodyC::IsGetEOS() const
  {
    // EOS if no file open
    if (!m_dvdFile)
      return true;
    
    // EOS if at last block
    if (m_byteCurrent >= (m_sizeData * DVD_VIDEO_LB_LEN))
      return true;
    
    return false;
  }

  bool DVDReadBodyC::Discard()
  {
    // Check we're not at the end of the stream
    if (IsGetEOS())
      return false;
    
    // Skip a byte
    m_byteCurrent++;
    
    return false;
  }

  bool DVDReadBodyC::GetAttr(const StringC &attrName, StringC &attrValue)
  {
    if (m_dvdVtsFile != NULL)
    {
      vtsi_mat_t *vtsi_mat = m_dvdVtsFile->vtsi_mat;
      video_attr_t &video_attr = vtsi_mat->vts_video_attr;
      
      if (attrName == "mpegversion")
      {
        attrValue = (video_attr.mpeg_version == 1 ? "MPEG2" : "MPEG1");
        return true;
      }

      if (attrName == "videoformat")
      {
        attrValue = (video_attr.video_format == 1 ? "PAL" : "NTSC");
        return true;
      }

      if (attrName == "aspectratio")
      {
        attrValue = (video_attr.display_aspect_ratio == 3 ? "16:9" : "4:3");
        return true;
      }

      if (attrName == "framesize")
      {
        UIntT height = 480;
        if(video_attr.video_format != 0) 
          height = 576;
        
        StringC width;
        switch(video_attr.picture_size)
        {
          case 0:
            width = "720x";
            break;
            
          case 1:
            width = "704x";
            break;

          case 2:
            width = "352x";
            break;

          case 3:
            width = "352x";
            height /= 2;
            break;
        }
        attrValue = width + height;
        return true;
      }
    }

    return DPPortBodyC::GetAttr(attrName, attrValue);
  }
  
  bool DVDReadBodyC::GetAttrList(DListC<StringC> &list) const
  {
    list.InsLast(StringC("mpegversion"));
    list.InsLast(StringC("videoformat"));
    list.InsLast(StringC("aspectratio"));
    list.InsLast(StringC("framesize"));
    return DPPortBodyC::GetAttrList(list);
  }

}
