# Copyright (C) 2003-14, OmniPerception Ltd.
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here

REQUIRES=RLog libGTK2
# Requirement for RLog is from using the relevant Ravl library
# rather than a direct dependency on the external library.
# Requirement for libGTK2 is both direct and from utilising
# RavlGUI and RavlGUI2D.

PACKAGE= Ravl/Face

MAINS =  markupTool.cc

LOCALHEADERS = ViewPage.hh ControlWin.hh

SOURCES = ViewPage.cc ControlWin.cc

PLIB = MarkupTool

USESLIBS = RavlGUI RavlFace RavlGUI2D GTK

PROGLIBS = RavlExtImgIO RavlImgIOJasper.opt RLog

