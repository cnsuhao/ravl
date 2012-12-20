// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
////////////////////////////////////////
//! rcsid="$Id$"
//! lib=RavlOS
//! file="Ravl/OS/Time/testDate.cc"
//! author="Charles Galambos"
//! docentry="Ravl.API.OS.Time"
//! userlevel=Develop

#include "Ravl/OS/DateRange.hh"
#include "Ravl/String.hh"
#include "Ravl/Stream.hh"
#include "Ravl/SysLog.hh"
//#include <stdlib.h>

using namespace RavlN;

int CheckDateRange();

int main()
{
  int lineno;
  cerr << "Starting DateC tests... \n";
  if ((lineno = CheckDateRange()) != 0) {
    cerr << "CheckDateRange(), Failed :" << lineno << "\n";
    return 1;
  }
  cerr << "Test passed.\n";
  return 0;
}

int CheckDateRange()
{
  // OK we want to get a date range for today in my time-zone
  DateRangeC today = DateRangeC::Today();

  // So when we display it we can display it in the local time-zone
  RavlDebug("Today where I am in LocalTime '%s'", today.Text(true).data());

  // Or its UTC equivalent
  RavlDebug("Today where I am in UTC '%s'", today.Text().data());

  DateRangeC yesterday = DateRangeC::Yesterday();
  RavlDebug("Yesterday in LocalTime %s", yesterday.Text(true).data());
  RavlDebug("Yesterday in UTC %s", yesterday.Text().data());

  RavlDebug("Last 3 days in LocalTime %s", DateRangeC::Days(3).Text(true).data());

  RavlDebug("Last week in LocalTime %s", DateRangeC::Days(7).Text(true).data());

  RavlDebug("Last 2 hours in LocalTime %s", DateRangeC::Hours(2).Text(true).data());

  RavlDebug("Last 10 minutes in LocalTime %s", DateRangeC::Minutes(10).Text(true).data());

  DateRangeC dr;
  RavlDebug("Default constructor date-range '%s'", dr.Text(true).data());

  return 0;
}
