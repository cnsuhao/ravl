# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2012, Charles Galambos
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
#! file="Ravl/Contrib/Audio/PortAudio/defs.mk"

REQUIRES = PortAudio

#DONOT_SUPPORT=VCPP

PACKAGE= Ravl/Audio

HEADERS= PortAudioIO.hh PortAudioFormat.hh

SOURCES= PortAudioIO.cc PortAudioFormat.cc

EXAMPLES= exPortAudio.cc 

PLIB=RavlPortAudio

MUSTLINK=RavlPortAudio.cc

USESLIBS=RavlIO RavlAudioUtil RavlAudioIO portAudio RavlThreads

PROGLIBS= RavlAudioFile.opt

EXTERNALLIBS=portAudio.def

