# This file is part of RAVL, Recognition And Vision Library
# This code may be redistributed under the terms of the GNU Lesser
# Copyright (C) 2005, Charles Galambos.
# see http://www.gnu.org/copyleft/lesser.html
# General Public License (LGPL). See the lgpl.licence file for details or
# file-header-ends-here

REQUIRES=MacOSX

PACKAGE=Ravl/MacOSX

HEADERS=BufferCVImage.hh QTFormat.hh QTVideoCaptureThread.h QTImageIPort.hh

SOURCES=BufferCVImage.cc QTFormat.cc QTVideoCaptureThread.mm  QTImageIPort.mm

PLIB=RavlMacOSXVideoCapture

MAINS=exQTCapture.cc

USESLIBS=RavlCore RavlOSIO RavlImageIO RavlDPDisplay ObjC OSXFoundation OSXQTKit OSXQuartzCore RavlMacOSXRunLoop

PROGLIBS=DynLink

MUSTLINK=LinkQTVideoCapture.cc
