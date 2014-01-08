# This file is part of RAVL, Recognition And Vision Library
# Copyright (C) 2008, OmniPerception Ltd.
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# Python is used under the terms of the Python License
# Copyright (C) 2001, 2002, 2003, 2004 Python Software Foundation; All Rights Reserved
# file-header-ends-here

REQUIRES = LibPython

PACKAGE = Ravl

HEADERS = Python.hh PythonObject.hh PythonException.hh PythonLock.hh PythonMainState.hh

SOURCES = Python.cc PythonObject.cc PythonException.cc PythonLock.cc PythonMainState.cc

PLIB = RavlPython

USESLIBS = RavlOS RavlCore RavlThreads Python 

PROGLIBS = RavlGUI

TESTEXES =

EXAMPLES = exPython.cc exPythonThreaded.cc exPythonMultipleInterpreters.cc exPyGTK.cc

EXTERNALLIBS = Python.def
