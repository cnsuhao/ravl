// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2011, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// Python is used under the terms of the Python License 
// Copyright (C) 2001, 2002, 2003, 2004 Python Software Foundation; All Rights Reserved
// file-header-ends-here
//////////////////////////////////////////////////////////////////
//! file = "Ravl/Contrib/Python/exPyGTK.cc"
//! lib=RavlPython
//! author = "Warren Moore"
//! docentry = "Ravl.API.Python"

#include "Ravl/GUI/Window.hh"
#include "Ravl/GUI/Button.hh"
#include "Ravl/GUI/Manager.hh"
#include "Ravl/GUI/LBox.hh"
#include "Ravl/Option.hh"
#include "Ravl/Python.hh"
#include "Ravl/PythonException.hh"
#include "Ravl/PythonObject.hh"

using namespace RavlGUIN;

bool GUICreateWindow(PythonC &python)
{
  try
  {
    static bool windowCreated = false;
    if (windowCreated)
      python.CallMethod("hello", "close");

    python.Run("hello = expygtk.HelloWorld()");
    windowCreated = true;
  }
  catch (PythonExceptionC &e)
  {
    std::cerr << "Python exception: " << e.Type() << " - " << e.Value() << std::endl;
  }

  return true;
}

bool CreateWindow(PythonC &python)
{
  std::cerr << "Queueing window creation on the GUI thread" << std::endl;

  Manager.QueueOnGUI(Trigger(&GUICreateWindow, python));

  return true;
}

bool GetValue(PythonC &python)
{
  std::cerr << "Calling HelloWorld.getX()\n";

  try
  {
    PythonObjectC obj = python.CallMethod("hello", "getX");
    if (obj.IsValid() && obj.IsInt())
    {
      std::cerr << "value = " << obj.Int() << std::endl;
    }
  }
  catch (PythonExceptionC &e)
  {
    std::cerr << "Python exception: " << e.Type() << " - " << e.Value() << std::endl;
  }

  return true;
}

int main(int argc, char *argv[])
{
  // Store the default Ctrl-C handler
  struct sigaction actionDefault;
  sigaction(SIGINT, NULL, &actionDefault);

  try
  {
    PythonC python(true);

    // Swap the Python KeyboardInterrupt handler for the default Ctrl-C handler
    sigaction(SIGINT, &actionDefault, NULL);

    python.AppendSystemPath(".");
    python.Import("expygtk");

    WindowC win(100, 100, "PyGTK");

    // Build and show the GUI
    win.Add(
              VBox
              (
                Button("Create window", CreateWindow, python) +
                Button("Get window value", GetValue, python)
              )
           );

    win.GUIShow();
  }
  catch (PythonExceptionC &e)
  {
    std::cerr << "Python exception: " << e.Type() << " - " << e.Value() << std::endl;
  	return -1;
  }
  
  // Start the GUI
  Manager.Start();
  std::cerr << "Finished... \n";
  return 0;
}
