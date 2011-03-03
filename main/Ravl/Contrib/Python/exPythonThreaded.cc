// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2008, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// Python is used under the terms of the Python License 
// Copyright (C) 2001, 2002, 2003, 2004 Python Software Foundation; All Rights Reserved
// file-header-ends-here
//////////////////////////////////////////////////////////////////
//! file = "Ravl/Contrib/Python/exPythonThreaded.cc"
//! lib = RavlPython
//! author = "Warren Moore"
//! docentry = "Ravl.API.Python"

#include "Ravl/Option.hh"
#include "Ravl/Python.hh"
#include "Ravl/PythonObject.hh"
#include "Ravl/PythonException.hh"
#include "Ravl/String.hh"
#include "Ravl/Threads/LaunchThread.hh"
#include "Ravl/OS/Date.hh"

using namespace RavlN;

void displayError(const char *errorMessage)
{
  cerr << "#### Error" << endl;
  cerr << "## " << errorMessage << endl;
}

void displayException(PythonExceptionC &e)
{
  cerr << "#### Error" << endl;
  cerr << "##  Type:  " << e.Type() << endl;
  cerr << "##  Value: " << e.Value() << endl;
  cerr << "##  Trace: " << endl << e.Trace() << endl;
}

bool queryGlobalThread(PythonC &python)
{
  try
  {
    int val = -1;
    while (val < 100)
    {
      PythonObjectC x = python.GetGlobal("x");
 //     cerr << "## Looking for global 'x': " << (x.IsValid() ? "Found" : "Not found") << endl;
      if (x.IsValid())
      {
        if (x.IsInt())
        {
          val = x.Int();
          cerr << "read(x = " << val << ")" << endl;
        }
        else
        {
          cerr << "#### Global 'x' is not an int" << endl;
          break;
        }
      }
      else
      {
        cerr << "#### Failed to find global 'x'" << endl;
        break;
      }
    }
  }
  catch (PythonExceptionC &e)
  {
    displayException(e);
  	return -1;
  }

  return true;
}

int main(int argc, char **argv)
{
  // Cmd line options
  OptionC opts(argc, argv);
  IntT threads = opts.Int("n", 10, "Number of threads");
  opts.Check();
  
  try 
  {
    // Initialise the python module
    PythonC python(true);

    cerr << "#### Initialising interpreter" << endl;
    if (!python.Initialised())
    {
      displayError("Failed to initialise interpreter");
      return -1;
    }

	  // Run a script from a string
	  cerr << "#### Initialise global 'x' (from string)" << endl;
	  const char* scriptInitialise = "x = 0";
    if (!python.Run(scriptInitialise))
    {
      cerr << "#### Failed to run script" << endl;
      return -1;
    }

    Sleep(1);

    for (int i = 0; i < threads; i++)
      LaunchThread(&queryGlobalThread, python);

    const char* scriptLoop = "\
for i in range(0, 100):\n\
  x += 1\n\
  print 'write(x = %d)' % x";

    cerr << "#### Calling loop script (from string)" << endl;
    if (!python.Run(scriptLoop))
    {
      cerr << "#### Failed to run script" << endl;
      return -1;
    }

    Sleep(1);
  }
  catch (PythonExceptionC &e)
  {
    displayException(e);
  	return -1;
  }

  return 0;
}
