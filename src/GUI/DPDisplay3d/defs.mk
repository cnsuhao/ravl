# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2008-14, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here

PACKAGE=Ravl/GUI

REQUIRES=libGL libGTK2
# Requirement for libGTK2 is from the use of RavlGUI and RavlGUI3D rather
# than a direct dependency.

HEADERS=DPDisplayView3d.hh DPWindow3d.hh DPWindow3dOPort.hh \
 DPWindow3dFormat.hh 

SOURCES=DPDisplayView3d.cc DPWindow3d.cc DPWindow3dOPort.cc \
 DPWindow3dFormat.cc DPDisplay3dConv.cc

PLIB=RavlDPDisplay3d

MUSTLINK= linkRavlDPDisplay3d.cc

USESLIBS=RavlIO RavlGUI RavlImageIO RavlGUI3D Ravl3DIO Ravl3D RavlImage

EXAMPLES=exDPWindow3d.cc

EHT=Ravl.API.GUI.Data_Display_3d.html
