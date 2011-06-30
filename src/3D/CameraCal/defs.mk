# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2001, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
#! rcsid="$Id$"
#! file="Ravl/3D/CameraCal/defs.mk"
PACKAGE=Ravl/3D
HEADERS=PinholeCamera.hh PinholeCamera0.hh PinholeCamera1.hh PinholeCamera2.hh PinholeCameraArray.hh
SOURCES=PinholeCamera0.cc PinholeCamera1.cc PinholeCamera2.cc PinholeCameraArray.cc
PLIB=RavlCameraCal
USESLIBS= RavlCore RavlImage RavlMath
EHT=Ravl.API.3D.Camera_Calibration.html
