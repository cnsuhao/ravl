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
#include "Ravl/DP/AttributeValueTypes.hh"

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
    m_title(title - 1),
    m_numFps(0.0),
    m_dvdReader(NULL),
    m_dvdVmgFile(NULL),
    m_dvdVtsFile(NULL),
    m_dvdPgc(NULL),
    m_dvdFile(NULL)
  {
    // Try to open the DVD device
    m_dvdReader = DVDOpen(m_device);
    if (!m_dvdReader)
    {
      cerr << "DVDReadBodyC::DVDReadBodyC unable to open DVD device (" << m_device << ")" << endl;
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
  
    // Check the angles
    if (tt_srpt->title[m_title].nr_of_angles > 1)
    {
      cerr << "DVDReadBodyC::DVDReadBodyC multiple angle DVDs not supported" << endl;
      Close();
      return;
    }
      
    // Get the title info
    UIntT vtsnum = tt_srpt->title[m_title].title_set_nr;
    UIntT ttnnum = tt_srpt->title[m_title].vts_ttn;
    ONDEBUG(cerr << "DVDReadBodyC::DVDReadBodyC VTS(" << vtsnum << ") title(" << ttnnum << ")" << endl;)

    // Get the VTS IFO
    m_dvdVtsFile = ifoOpen(m_dvdReader, vtsnum);
    if(!m_dvdVtsFile)
    {
      cerr << "DVDReadBodyC::DVDReadBodyC unable to load IFO for VTS (" << vtsnum << ")" << endl;
      Close();
      return;
    }
    
    // Get the program chain info
    m_dvdPgc = m_dvdVtsFile->vts_pgcit->pgci_srp[0].pgc;
    UIntT numCells = m_dvdPgc->nr_of_cells;
    if (numCells == 0)
    {
      cerr << "DVDReadBodyC::DVDReadBodyC no cells found for title (" << ttnnum << ")" << endl;
      Close();
      return;
    }
    ONDEBUG(cerr << "DVDReadBodyC::DVDReadBodyC cells(" << numCells << ")" << endl;)

    // Get the FPS
    m_numFps = 0.0;
    dvd_time_t titleTime = m_dvdPgc->playback_time;
    if (titleTime.hour != 0 &&  titleTime.minute != 0 && titleTime.second != 0)
    {
      if ((titleTime.frame_u & 0xc0) >> 6 == 1)
        m_numFps = 25.0;
      if ((titleTime.frame_u & 0xc0) >> 6 == 3)
        m_numFps = 29.97;
    }
    ONDEBUG(cerr << "DVDReadBodyC::DVDReadBodyC fps(" << m_numFps << ")" << endl;)

    // Create the cell data table, if no tmap data found
    RealT playTime = 0.0;
    m_cellTable = SArray1dC< Tuple3C< Int64T, StreamPosT, bool> >(numCells);
    for (UIntT i = 0; i < numCells; i++)
    {
      m_cellTable[i].Data1() = (Int64T)Floor(playTime * m_numFps);
      m_cellTable[i].Data2() = m_dvdPgc->cell_playback[i].first_sector;
      m_cellTable[i].Data3() = m_dvdPgc->cell_playback[i].stc_discontinuity;
      ONDEBUG(cerr << "DVDReadBodyC::DVDReadBodyC cell(" << i << ")\tframe(" << m_cellTable[i].Data1() << ")\tsector(" << hex << m_cellTable[i].Data2() << dec << ")\tdiscontinuity(" << (m_cellTable[i].Data3() ? "y" : "n") << ")" << endl;)

      // Get the cell payback info
      dvd_time_t cellTime = m_dvdPgc->cell_playback[i].playback_time;
      playTime += GetTime(cellTime);
    }

    // If found, create the tmap data table
    if (m_dvdVtsFile->vts_tmapt)
    {
      // Get the number of TMAPs (should be one per title, even if empty)
      vts_tmapt_t *tmapt = m_dvdVtsFile->vts_tmapt;
      UIntT numTmaps = tmapt->nr_of_tmaps;
      ONDEBUG(cerr << "DVDReadBodyC::DVDReadBodyC tmaps(" << numTmaps << ")" << endl;)
      
      // Select the correct TMAP for the title
      if (numTmaps > 0 && ttnnum <= numTmaps)
      {
        // Get the number of entries for the title's TMAP (may be zero)
        UIntT timeUnit = tmapt->tmap[ttnnum - 1].tmu;
        UIntT numTmapEntries = tmapt->tmap[ttnnum - 1].nr_of_entries;
        ONDEBUG(cerr << "DVDReadBodyC::DVDReadBodyC time unit(" << timeUnit << ") tmap entries(" << numTmapEntries << ")" << endl;)
        
        // If we have entries, fill the sector table with them
        Int64T cellOffset = 0;
        if (numTmapEntries > 0)
        {
          playTime = 0.0;
          m_tmapTable = SArray1dC< Tuple3C< Int64T, StreamPosT, bool> >(numTmapEntries);
          for (UIntT i = 0; i < numTmapEntries; i++)
          {
            playTime += timeUnit;
            m_tmapTable[i].Data1() = (Int64T)Floor(playTime * m_numFps) + cellOffset;
            m_tmapTable[i].Data2() = tmapt->tmap[ttnnum - 1].map_ent[i] & 0x7fffffff;
            m_tmapTable[i].Data3() = (tmapt->tmap[ttnnum - 1].map_ent[i] >> 31) != 0;
            ONDEBUG(cerr << "DVDReadBodyC::DVDReadBodyC tmap entry(" << i << ")\tframe(" << m_tmapTable[i].Data1() << ")\tsector(" << hex <<  m_tmapTable[i].Data2() << dec << ")\tdiscontinuity(" << (m_tmapTable[i].Data3() ? "y" : "n") << ")" << endl;)
            
            // If a discontinuity is detected, reset the playtime and find the offset
            if (m_tmapTable[i].Data3())
            {
              playTime = 0;
              UIntT cellCount = 1;
              while (cellCount < (m_cellTable.Size() - 1) && m_tmapTable[i].Data2() > m_cellTable[cellCount].Data2())
              {
                cellCount++;
              }
              cellOffset = m_cellTable[cellCount].Data1();
            }
          }
        }
      }
    }

    // Open the title
    m_dvdFile = DVDOpenFile(m_dvdReader, vtsnum, DVD_READ_TITLE_VOBS);
    if (!m_dvdFile)
    {
      cerr << "DVDReadBodyC::DVDReadBodyC unable to open title VOB VTS(" << vtsnum << ") for title (" << ttnnum << ")" << endl;
      Close();
      return;
    }

    // Start at the first frame
    SeekFrame(0);
    
    BuildAttributes();
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
    m_numFps = 0.0;
    
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
    return 0;
  }

  bool DVDReadBodyC::Seek(UIntT off)
  {
    return false;
  }

  UIntT DVDReadBodyC::Tell() const
  {
    return ((UIntT)(-1));
  }

  UIntT DVDReadBodyC::Size() const
  {
    return ((UIntT)(-1));
  }

  bool DVDReadBodyC::Seek64(StreamPosT off)
  {
    return false;
  }
  
  StreamPosT DVDReadBodyC::Tell64() const
  {
    return -1;
  }
  
  StreamPosT DVDReadBodyC::Size64() const
  {
    return -1;
  }

  bool DVDReadBodyC::IsGetEOS() const
  {
    return true;
  }

  bool DVDReadBodyC::Discard()
  {
    return false;
  }

  StreamPosT DVDReadBodyC::SeekFrame(const StreamPosT frame)
  {
    cerr << "DVDReadBodyC::SeekFrame frame(" << frame << ")" << endl;
    // Find the closest sector
    UIntT sectorCount = 0;
    while (sectorCount < m_cellTable.Size() && frame > m_cellTable[sectorCount].Data1())
    {
      sectorCount++;
    }
    
    // Check we have a valid sector table entry
    if (sectorCount >= m_cellTable.Size())
    {
      cerr << "DVDReadBodyC::SeekFrame overflow" << endl;
      m_curSector = sectorCount = 0;
    }
    
    // Store the sector position
    m_curSector = m_cellTable[sectorCount].Data2();
    cerr << "DVDReadBodyC::SeekFrame sector(" << m_curSector << ") frame(" << m_cellTable[sectorCount].Data1() << ")" << endl;
    
    // Reset the byte pointer
    m_curByte = 0;

    return m_cellTable[sectorCount].Data1();
  }
  
  Int64T DVDReadBodyC::GetTime(dvd_time_t time)
  {
    Int64T s = 0;
    s += ((time.hour >>   4) & 0x0f) * 36000;
    s += ((time.hour       ) & 0x0f) *  3600;
    s += ((time.minute >> 4) & 0x0f) *   600;
    s += ((time.minute     ) & 0x0f) *    60;
    s += ((time.second >> 4) & 0x0f) *    10;
    s += ((time.second     ) & 0x0f)        ;
    return s;
  }
  
  bool DVDReadBodyC::GetAttr(const StringC &attrName, StringC &attrValue)
  {
    RavlAssertMsg(m_dvdVtsFile != NULL, "DVDReadBodyC::GetAttr requires a valid VTS file");
    
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
        IntT height = 480;
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
          
          default:
            height = 0;
        }
        if (height != 0)
        {
          attrValue = width + StringC(height);
          return true;
        }
      }
      
      if (attrName == "duration")
      {
        RavlAssertMsg(m_dvdPgc != NULL, "DVDReadBodyC::GetAttr requires a valid PGC");
        
        dvd_time_t time = m_dvdPgc->playback_time;
        if (time.hour != 0 &&  time.minute != 0 && time.second != 0)
        {
          attrValue.form("%02x:%02x:%02x", time.hour, time.minute, time.second);
          return true;
        }
      }

      if (attrName == "framerate")
      {
        RavlAssertMsg(m_numFps != 0, "DVDReadBodyC::GetAttr FPS not valid");
        
        attrValue.form("%2.2f", m_numFps);
        return true;
      }

      if (attrName == "frames")
      {
        RavlAssertMsg(m_numFps != 0, "DVDReadBodyC::GetAttr FPS not valid");
        
        dvd_time_t time = m_dvdPgc->playback_time;
        Int64T frames = (Int64T)Floor(((RealT)GetTime(time)) * m_numFps);
        attrValue = StringC(frames);
        return true;
      }
    }

    return DPPortBodyC::GetAttr(attrName, attrValue);
  }

  //: Register stream attributes.

  void DVDReadBodyC::BuildAttributes()
  {
    RegisterAttribute(AttributeTypeStringC("mpegversion", "MPEG version", true, false));
    RegisterAttribute(AttributeTypeStringC("videoformat", "Video format", true, false));
    RegisterAttribute(AttributeTypeStringC("aspectratio", "Aspect ratio", true, false));
    RegisterAttribute(AttributeTypeStringC("framesize",   "Frame size",   true, false));
    RegisterAttribute(AttributeTypeStringC("duration",    "Duration",     true, false));
    RegisterAttribute(AttributeTypeStringC("framerate",   "Frame rate",   true, false));
    RegisterAttribute(AttributeTypeStringC("frames",      "Total frames", true, false));
  }

}
