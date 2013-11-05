# This file is part of OmniSoft,
# Copyright (C) 2003, OmniPerception Ltd.
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
#    * MAINS A list of programs to compile.
#    * SOURCES Source files to compile.
#    * HEADERS Header files needed for this source code.
#    * PLIB Name of library to create.
#    * USESLIBS Libraries needed to compile source code.
#    * NESTED Subdirectories to compile.
#! file="OmniSoft/Applications/WhoFIT/defs.mk"

REQUIRES=RLog

PACKAGENAME = RavlFace

PACKAGE= Ravl/Face

MAINS =  markupTool.cc

LOCALHEADERS = ViewPage.hh ControlWin.hh

SOURCES = ViewPage.cc ControlWin.cc

PLIB = MarkupTool

SUMMARY_LIB=Ravl

USESLIBS = RavlGUI RavlFace RavlGUI2D

PROGLIBS = RavlExtImgIO RavlImgIOJasper.opt

