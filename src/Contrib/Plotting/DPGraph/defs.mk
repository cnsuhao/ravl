# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2001-14, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here

DONOT_SUPPORT=VCPP

REQUIRES=libGuppi libGTK2
# Requirement for libGTK2 stems from using RavlGUI rather than being
# a direrct dependency.

PACKAGE=Ravl/Plot

HEADERS= DPGraphWindow.hh DPGraphWindowOPort.hh DPGraphWindowFormat.hh

SOURCES= DPGraphWindow.cc DPGraphWindowOPort.cc DPGraphWindowFormat.cc

PLIB=RavlDPGraph

MUSTLINK=RavlDPGraphWindow.cc

USESLIBS=RavlIO RavlGUI RavlPlot RavlGuppi

EXAMPLES= exDPGraphWindow.cc

EHT=Ravl.API.GUI.Data_Plotting.html
