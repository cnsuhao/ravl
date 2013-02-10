#include "Ravl/DP/ProcessPlayList.hh"
#include "Ravl/XMLFactory.hh"
#include "Ravl/OS/Filename.hh"
#include "Ravl/DP/SequenceIO.hh"
#include "Ravl/DP/Plug.hh"
#include "Ravl/DP/TypeInfo.hh"

namespace RavlN {

  //! Constructor
  ProcessPlayListC::ProcessPlayListC()
   : m_verbose(false)
  {}

  //! Process list.
  bool ProcessPlayListC::ProcessList()
  {

    SArray1dC<EditSpecC> playList = m_playList.Edits();

    for(unsigned i = 0;i < playList.Size();i++) {
      Process(playList[i]);
    }

    return true;
  }

  //! Process a single file.
  bool ProcessPlayListC::Process(const SubSequenceSpecC &seqSpec) {

    FilenameC fn = seqSpec.Filename();
    FilenameC pc = fn.PathComponent();
    FilenameC nc = fn.NameComponent();
    FilenameC bc = fn.BaseNameComponent();

    FilenameC outputName = m_outputTemplate;
    outputName.gsub("%p",pc);
    outputName.gsub("%n",nc);
    outputName.gsub("%b",bc);

    DPIPortBaseC isource;
    DPIPlugBaseC iplug;
    m_process.GetIPlug(m_inputName,iplug);

    DPSeekCtrlC seekControl;
    if(!OpenISequenceBase(isource,seekControl,seqSpec.Filename(),seqSpec.FileFormat(),iplug.InputType(),m_verbose)) {
      RavlError("Failed to open input '%s'",seqSpec.Filename().c_str());
      return false;
    }

    if(!iplug.ConnectPort(isource)) {
      RavlError("Failed to connect port.");
      return false;
    }

    DPIPortBaseC pullPort;
    if(!m_process.GetIPort(m_ouputName,pullPort)) {
      RavlError("Failed to get output");
      return false;
    }

    DPOPortBaseC pushPort;

    DPSeekCtrlC seekControlO;
    if(!OpenOSequenceBase(pushPort,seekControl,outputName,"",pullPort.InputType(),m_verbose)) {
      RavlError("Failed to open output '%s'",seqSpec.Filename().c_str());
      return false;
    }

    DPTypeInfoC theType = TypeInfo(pullPort.InputType());
    if(!theType.IsValid()) {
      RavlError("Type '%s' unknown, can't complete process.");
      return false;
    }

    IndexC ff = seqSpec.FirstFrame();
    if(ff != 0) {
      bool done= false;
      if(seekControl.IsValid())
        done = seekControl.Seek64(ff.V());
      if(!done) {
        // Do it the hard way.
        for(IndexC i =0;i < ff;i++)
          pullPort.Discard();
      }
    }

    if(seqSpec.LastFrame() == RavlConstN::maxInt) {
      while(theType.Move(pullPort,pushPort,32) != 0) ;
    } else {
      // FIXME:- THe sequence length of the output and input need not match.
      theType.Move(pullPort,pushPort,seqSpec.Range().Size());
    }

    return true;
  }

}
