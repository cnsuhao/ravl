// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2010, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html

%include "Ravl/Swig2/Types.i"
%include "Ravl/Swig2/String.i"

%{
#include "Ravl/OS/SysLog.hh"
%}

namespace RavlN
{

  bool SysLogOpen(const StringC &name,bool logPid = false,bool sendStdErr = true,bool stdErrOnly = false,int facility = -1,bool logFileLine = false);
  //: Open connection to system logger.
  // Facility is set to 'LOG_USER' by default. <br>
  // If logPid is true the processes id will be recorded in the log. <br>
  // If sendStdErr is set the messages will also be send the standard error channel.
  
  
}
