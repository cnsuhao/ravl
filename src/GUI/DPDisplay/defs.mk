# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2001-11, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
#! file="Ravl/GUI/DPDisplay/defs.mk"

PACKAGE=Ravl/GUI

HEADERS=DPWindow.hh DPDisplayObj.hh DPDisplayImage.hh DPWindowOPort.hh \
 DPWindowFormat.hh DPDisplayView.hh DPDisplayPoints.hh DPDisplayPolygon.hh

SOURCES=DPWindow.cc DPDisplayObj.cc DPDisplayImage.cc DPWindowOPort.cc \
 DPWindowFormat.cc DPDisplayView.cc DPDisplayPoints.cc DPDisplayPolygon.cc

PLIB=RavlDPDisplay

SUMMARY_LIB=Ravl

MUSTLINK=RavlGUIDisplay.cc

REQUIRES=libGTK2

USESLIBS=RavlIO RavlGUI RavlImageIO RavlGUI2D RavlImage GTK

EXAMPLES=exDPWindow.cc

EHT=Ravl.API.GUI.Data_Display.html
