// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlOS
//! author="Charles Galambos"
//! file="Ravl/OS/Misc/SysLog.cc"

#include "Ravl/OS/SysLog.hh"
#include "Ravl/StrStream.hh"
#include "Ravl/Calls.hh"
#include "Ravl/MTLocks.hh"

#if RAVL_COMPILER_VISUALCPP
#pragma warning ( disable : 4244 )
#pragma warning ( disable : 4267 )
#include <stdarg.h>
#endif

#if RAVL_OS_POSIX
#include <syslog.h>
#if RAVL_HAVE_UNISTD_H
#include <unistd.h>
#endif
#endif

#include <stdio.h>

namespace RavlN {
  
  static StringC syslog_ident("NoName");
  static bool syslog_Open = false;
  static bool syslog_StdErrOnly = false;
  static bool syslog_StdErr = true;
  static bool syslog_pid = false;
  static bool syslog_fileline = false;
  static int localLevel = 8;
  static int syslogLevel = 8;
  
  typedef void (*SyslogFunc)(SysLogPriorityT level,const char *message,unsigned lineno,const char *filename);
  
  static SyslogFunc syslogRedirect = 0;
  
  //: Register function to redirect log messages.
  // Calling with a null function pointer restores the default behavour.
  
  bool SysLogRedirect(void (*logFunc)(SysLogPriorityT level,const char *message,unsigned lineno,const char *filename)) {
    MTWriteLockC lockWrite(2); // Be careful in multi-threaded programs.
    syslogRedirect = logFunc;
    return true;
  }
  
  //: Get the name of the current application.
  
  const StringC &SysLogApplicationName()
  { return syslog_ident; }
  
  //: Open connection to system logger.
  // Facility is set to 'LOG_USER' by default. <br>
  // If logPid is true the processes id will be recorded in the log. <br>
  // If sendStdErr is set the messages will also be send the standard error channel.
  
  bool SysLogOpen(const StringC &name,bool logPid,bool sendStdErr,bool stdErrOnly,int facility,bool logFileLine) {
    syslog_ident = name;
    syslog_Open = true;
    syslog_StdErrOnly = stdErrOnly;
    syslog_pid = logPid;
    if(stdErrOnly)
      syslog_StdErr = true;
    syslog_StdErr = sendStdErr;
    syslog_fileline = logFileLine;
#if RAVL_OS_POSIX
    if(facility == -1)
      facility = LOG_USER;
    int options = 0;
    if(logPid)
      options |= LOG_PID;
#if !RAVL_OS_SOLARIS
    if(sendStdErr)
      options |= LOG_PERROR;
#endif
    openlog(syslog_ident,options,facility);
#endif
    return true;
  }
  
  //: Close connection to system logger.
  // The call of this function is optional.
  
  bool SysLogClose() {
#if RAVL_OS_POSIX
    closelog();
    syslog_Open = false;
#endif
    return true;
  }
  
#if RAVL_OS_POSIX
  // Mapping to syslog levels, though they should be identical.
  static const int messageTypes[8] = { 
    LOG_EMERG,
    LOG_ALERT,
    LOG_CRIT,
    LOG_ERR,
    LOG_WARNING,
    LOG_NOTICE,
    LOG_INFO,
    LOG_DEBUG
  };
#endif
  
  static bool LogMessage(const char *message,int priority,unsigned lineno =0,const char *filename = 0) {
    MTWriteLockC lockWrite(2); // Be careful in multi-threaded programs.
    if(syslogRedirect != 0) {
      // Avoid possible deadlocks when handling redirected messages
      // (If the processes of logging causes a log message to be generated.)
      SyslogFunc syslogRedirectFunc = syslogRedirect;
      lockWrite.Unlock();
      syslogRedirectFunc((SysLogPriorityT)priority,message,lineno,filename);
      return true;
    }
#if RAVL_OS_POSIX
    if(syslog_StdErrOnly) {
      if(priority <= localLevel) {
	std::cerr << syslog_ident;
	if(syslog_fileline)
	  std::cerr << filename << ':' << lineno << ' ';
	if(syslog_pid)
	  std::cerr << "[" << getpid() << "]";
	std::cerr << ":" << message << endl;
      }
    } else {
      if(priority < syslogLevel) {
        if(syslog_fileline) {
          syslog(priority,"%s:%u %s",filename,lineno,message);
        } else {
          syslog(priority,"%s",message);
        }
      } else {
	if(syslog_StdErr && priority <= localLevel) {
	  std::cerr << syslog_ident;
	  if(syslog_fileline)
	    std::cerr << filename << ':' << lineno << ' ';
	  if(syslog_pid)
	    std::cerr << "[" << getpid() << "]";
	  std::cerr << ":" << message << endl;
	}
      }
    }
#else
    if(syslog_fileline)
      std::cerr << filename << ':' << lineno << ' ';
    if(priority < localLevel)
      std::cerr << syslog_ident << ":" << message << endl;
#endif
    std::cerr << std::flush;
    return true;
  }
  
  //: Send a message to the log file
  // Usage: <br>
  // SysLog(SYSLOG_DEBUG) << "Send message to log";
  
  OStreamC SysLog(SysLogPriorityT priority,unsigned lineno,const char *filename) {
    if(!syslog_Open)
      SysLogOpen("NoName",true,true);
#if RAVL_OS_POSIX
    int pri = LOG_DEBUG;
    if(priority >= 0 && priority < 8)
      pri = messageTypes[(int) priority];
#else
    int pri = (int) priority;
#endif
    return StrOStreamC(Trigger(LogMessage,"",pri,lineno,filename));
  }

  //: Send a message to the log file
  // Usage: <br>
  // SysLog(SYSLOG_DEBUG,"msg",args...);
  
  void SysLog(SysLogPriorityT priority,const char *format,...) {
    const int formSize = 4096;
    va_list args;
    va_start(args,format);
    char buff[formSize];
    int x;
#if RAVL_COMPILER_VISUALCPPNET_2005
    if((x = vsprintf_s(buff,formSize,format,args)) < 0)
      std::cerr << "WARNING: SysLog(...), String truncated!! \n";
#elif RAVL_COMPILER_VISUALCPP
    if((x = _vsnprintf(buff,formSize,format,args)) < 0)
      std::cerr << "WARNING: SysLog(...), String truncated!! \n";
#else
    if((x = vsnprintf(buff,formSize,format,args)) < 0)
      std::cerr << "WARNING: SysLog(...), String truncated!! \n";
#endif
    va_end(args);
    
    LogMessage(buff,priority);
  }

  void SysLog(SysLogPriorityT priority,unsigned lineno,const char *filename,const char *format ...)
  {
    const int formSize = 4096;
    va_list args;
    va_start(args,format);
    char buff[formSize];
    int x;
#if RAVL_COMPILER_VISUALCPPNET_2005
    if((x = vsprintf_s(buff,formSize,format,args)) < 0)
      std::cerr << "WARNING: SysLog(...), String truncated!! \n";
#elif RAVL_COMPILER_VISUALCPP
    if((x = _vsnprintf(buff,formSize,format,args)) < 0)
      std::cerr << "WARNING: SysLog(...), String truncated!! \n";
#else
    if((x = vsnprintf(buff,formSize,format,args)) < 0)
      std::cerr << "WARNING: SysLog(...), String truncated!! \n";
#endif
    va_end(args);

    LogMessage(buff,priority,lineno,filename);
  }


  //: Set the level of messages to send to the system.
  // Only messages with a priority lower than 'level' we be sent.
  // This can be used to avoid flooding the log system
  // with debug messages for example.
  
  bool SysLogLevel(SysLogPriorityT level) 
  { 
    int pri = (int) level;
    if(pri < 0)
      pri = 0;
    else if(pri > 7)
      pri = 7;
#if RAVL_OS_POSIX
    syslogLevel = messageTypes[pri];
#else
    syslogLevel = pri;
#endif
    return true;
  }
  
  //: Set the level of messages to send to standard error.
  // This controls the level of messages to send to
  // cerr. <br>
  // Only messages with a priority lower than 'level' we be sent.
  
  bool SysLogLevelStdErr(SysLogPriorityT level) { 
    int pri = (int) level; 
    if(pri < 0)
      pri = 0;
    else if(pri > 7)
      pri = 7;    
#if RAVL_OS_POSIX
    localLevel = messageTypes[pri];
#else
    syslogLevel = pri;
#endif
    return true;
  }
  
  
}
