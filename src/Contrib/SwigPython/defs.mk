# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2006, OmniPerception Ltd.
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here

REQUIRES=LibPython Swig SwigPython

PACKAGE=Ravl/Swig

HEADERS=

SOURCES=Ravl.i

PLIB=RavlPythonSwig

SINGLESO=_RavlPython

USESLIBS=RavlDPDisplay.opt RavlCore RavlMath RavlImage RavlImageIO Python RavlNet RavlOSIO

PROGLIBS=DynLink

SWIGOPTS=-python

AUXDIR=/share/Python

AUXFILES=Ravl.py
