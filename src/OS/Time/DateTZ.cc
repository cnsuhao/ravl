
#undef _ANSI_SOURCE
#undef _POSIX_C_SOURCE

#include "Ravl/config.h"
#include "Ravl/SysLog.hh"
#if RAVL_OS_SOLARIS
#define _POSIX_PTHREAD_SEMANTICS 1
#define _REENTRANT 1
//#define __STDC__ 0
#include <time.h>
#endif

#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE 1
#endif

#if RAVL_OS_WIN32
// Turn off deprecated warnings for now, they're not deprecated on other platforms
// may introduce more platform specific fixes later.
#define _CRT_SECURE_NO_DEPRECATE 1
#include <sys/timeb.h>
#endif

#include <time.h>

#ifdef RAVL_OS_MACOSX
extern "C" {
  char *asctime_r(const struct tm *, char *);
  char *ctime_r(const time_t *, char *);
  struct tm *gmtime_r(const time_t *, struct tm *);
  struct tm *localtime_r(const time_t *, struct tm *);
}
#endif

#include "Ravl/Exception.hh"
#include "Ravl/OS/Date.hh"
#include "Ravl/String.hh"
#include "Ravl/StrStream.hh"

#include <stdlib.h>
#include <stdio.h>

#if RAVL_HAVE_WIN32_THREADS
#include <windows.h>
#endif

#if !RAVL_COMPILER_VISUALCPP
#include <sys/time.h>
#include <sys/timeb.h>
#include <unistd.h>
#else
#include <string.h>

#include <sys/types.h>

char *ctime_r(const time_t *s,char *buff) {
  strcpy(buff,ctime(s));
  return buff;
}

struct tm *localtime_r( const time_t *timer,struct tm *b) {
  memcpy(b,localtime(timer),sizeof(tm));
  return b;
}

extern int sleep(int x); // A hack, I hope it doesn't cause problems later...

#endif

#include "Ravl/OS/DateTZ.hh"
#include "Ravl/SysLog.hh"
#include "Ravl/BinStream.hh"

namespace RavlN {

  //: Default constructor
  DateTZC::DateTZC()
   : m_timeZoneOffsetMinutes(0)
  {}

  //: Construct from a real in seconds.
  DateTZC::DateTZC(RealT val)
   : m_localTime(val),
     m_timeZoneOffsetMinutes(0)
  {}

  //: Construct from a date and a timezone offset
  DateTZC::DateTZC(const DateC &localTime,IntT timezoneOffsetMinutes)
   : m_localTime(localTime),
     m_timeZoneOffsetMinutes(timezoneOffsetMinutes)
  {}

  //: Constructor.
  //!param: year - Year (Must be from 1901 to 2038 inclusive)
  //!param: month - Month in year 1-12 (Unlike mktime, which goes 0 to 11)
  //!param: day - Day in the month 1 to 31
  //!param: hour - 0 to 23
  //!param: min - 0 to 59
  //!param: sec - 0 to 59
  //!param: usec - 1000000's of a second.
  //!param: useLocalTimeZone - When true assume parameters are in the local time zone and convert to UTC
  DateTZC::DateTZC(IntT year,IntT month,IntT day,IntT hour,IntT min,IntT sec,IntT usec,IntT timezoneOffsetMinutes)
   : m_localTime(year,month,day,hour,min,sec,usec,false),
     m_timeZoneOffsetMinutes(timezoneOffsetMinutes)
  {}

  //: Get the time now

  DateTZC DateTZC::Now()
  { return DateTZC(DateC::NowLocal(),DateC::TimeZoneOffset().TotalSeconds()); }

  //: Generate date from ISO8601 string.
  // Note this may not support all variants, if the string fails to parse and exception will be thrown.
  DateTZC DateTZC::FromISO8601String(const StringC &dataString)
  {
    const StringC &work = dataString;
    int at = 0;
    UIntT year = 0;
    UIntT month = 1;
    UIntT day = 1;
    UIntT hour = 0;
    UIntT min = 0;
    UIntT sec = 0;
    UIntT usec = 0;
    IntT tzOffset = 0;

    //2012-01-30 10:00Z
    //2012-01-30T10:00Z
    //2012-01-30T10:00:00Z

    int elems = sscanf(&work[at],"%4u-%2u-%2u",&year,&month,&day);
    if(elems != 3) {
      RavlDebug("Failed to parse date from: '%s' ",work.data());
      throw ExceptionOperationFailedC("Parse error in date.");
    }
    if(work.Size() <= 10) {
      // No timezone, so must be local time.
      tzOffset = DateC::TimeZoneOffset().TotalSeconds();
      return DateTZC(year,month,day,hour,min,sec,0,tzOffset);
    }
    at += 10;
    if(work[at] != ' ' && work[at] != 'T') {
      RavlError("Invalid time %s ",work.c_str());
      throw ExceptionOperationFailedC("Parse error in date.");
    }
    at++;

    //RavlDebug("Scanning time:%s ",work.c_str());
    double secf;
    elems = sscanf(&work[at],"%2u:%2u:%lf",&hour,&min,&secf);
    if(elems != 3) {
      RavlDebug("Failed to parse time from: '%s' ",work.data());
      throw ExceptionOperationFailedC("Parse error in time.");
    }
    sec = secf;
    usec = RavlN::Round((secf - floor(secf)) * 1000000.0);
    at += 8;
    if(work[at] == '.') {
      at++;
      // Fractional seconds ?
      for(;at < work.Size();at++) {
        if(!isdigit(work[at]))
          break;
      }
    }

    if(work[at] == 0) {
      // No timezone, so must be local time.
      tzOffset = DateC::TimeZoneOffset().TotalSeconds();
      return DateTZC(year,month,day,hour,min,sec,usec,tzOffset);
    }

    if(work[at] == 'Z') {
      // UTZ timezone
      return DateTZC(year,month,day,hour,min,sec,usec);
    }
    if(work[at] != '+' && work[at] != '-') {
      RavlError("Invalid timezone %s ",work.c_str());
      throw ExceptionOperationFailedC("Invalid timezone");
    }
    bool isNeg = work[at] == '-';
    at++;

    int tzHours = 0;
    int tzMinutes = 0;
    elems = sscanf(&work[at],"%2u",&tzHours);
    if(work.Size()-at > 2) {
      at += 2;
      if(work[3]==':')
        at++;
      elems = sscanf(&work[at],"%2u",&tzMinutes);
    }

    tzOffset = tzHours * 60 + tzMinutes;
    if(isNeg) tzOffset *= -1;
    return DateTZC(year,month,day,hour,min,sec,usec,tzOffset);
  }

  //: Format as ISO8601 string
  StringC DateTZC::ISO8601() const
  {
    struct tm b;

    time_t s = (time_t) m_localTime.TotalSeconds();
#if !RAVL_COMPILER_VISUALCPP
    gmtime_r(&s,&b);
#else
    // VC++ does not support asctime_r or gmtime_r so use the non-thread-safe versions
    // in lieu os anythings else
    // In VC++ the result is stored in thread local storage, so this should
    // be thread safe.
    b = *gmtime(&s);
#endif
    StringC buf;

    double sec = (double) b.tm_sec + (m_localTime.USeconds() / 1000000.0f);
    if(m_timeZoneOffsetMinutes == 0) {
      if(b.tm_sec < 10) {
        buf.form("%04u-%02u-%02uT%02u:%02u:0%1fZ",b.tm_year + 1900,b.tm_mon+1,b.tm_mday,b.tm_hour,b.tm_min,sec);
      } else {
        buf.form("%04u-%02u-%02uT%02u:%02u:%2fZ",b.tm_year + 1900,b.tm_mon+1,b.tm_mday,b.tm_hour,b.tm_min,sec);
      }
      return buf;
    }
    int absOff;
    char signChar;
    if(m_timeZoneOffsetMinutes < 0) {
      absOff = -m_timeZoneOffsetMinutes;
      signChar = '-';
    } else {
      absOff = m_timeZoneOffsetMinutes;
      signChar = '+';
    }
    int tzHour = absOff/60;
    int tzMin = absOff%60;
    if(tzMin == 0) {
      if(b.tm_sec < 10) {
        buf.form("%04u-%02u-%02uT%02u:%02u:0%1f%c%02u",b.tm_year + 1900,b.tm_mon+1,b.tm_mday,b.tm_hour,b.tm_min,sec,signChar,tzHour);
      } else {
        buf.form("%04u-%02u-%02uT%02u:%02u:%0f%c%02u",b.tm_year + 1900,b.tm_mon+1,b.tm_mday,b.tm_hour,b.tm_min,sec,signChar,tzHour);
      }
      return buf;
    }

    if(b.tm_sec < 10) {
      buf.form("%04u-%02u-%02uT%02u:%02u:%02u%c0%1f:%20u",b.tm_year + 1900,b.tm_mon+1,b.tm_mday,b.tm_hour,b.tm_min,sec,signChar,tzHour,tzMin);
    } else {
      buf.form("%04u-%02u-%02uT%02u:%02u:%02u%c%2%f:%20u",b.tm_year + 1900,b.tm_mon+1,b.tm_mday,b.tm_hour,b.tm_min,sec,signChar,tzHour,tzMin);
    }
    return buf;
  }


  ostream &operator <<(ostream &strm,const DateTZC &date)
  {
    strm << date.ISO8601();
    return strm;
  }
  //: Stream operator.

  istream &operator >>(istream &strm,DateTZC &date)
  {
    StringC str;
    strm >> str;
    date = DateTZC::FromISO8601String(str);
    return strm;
  }
  //: Stream operator.

  BinOStreamC &operator <<(BinOStreamC &strm,const DateTZC &date)
  {
    ByteT version = 1;
    strm << version;
    strm << date.LocalTime() << (Int32T) date.TimeZoneOffsetMinutes();
    return strm;
  }
  //: Stream operator.

  BinIStreamC &operator >>(BinIStreamC &strm,DateTZC &date)
  {
    ByteT version = 1;
    strm >> version;
    if(version != 1)
      throw ExceptionUnexpectedVersionInStreamC("DateTZC");
    Int32T tzOffset = 0;
    DateC localTime;
    strm >> localTime >> tzOffset;
    date = DateTZC(localTime,tzOffset);
    return strm;
  }
  //: Stream operator.


}
