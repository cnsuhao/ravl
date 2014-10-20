# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2013-14, Charles Galambos
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here

REQUIRES = PulseAudioSimple

#DONOT_SUPPORT=VCPP

PACKAGE= Ravl/Audio

HEADERS= PulseAudioIO.hh PulseAudioFormat.hh

SOURCES= PulseAudioIO.cc PulseAudioFormat.cc

EXAMPLES= exPulseAudio.cc 

PLIB=RavlPulseAudio

MUSTLINK=RavlPulseAudio.cc

USESLIBS=RavlIO RavlAudioUtil RavlAudioIO PulseAudioSimple RavlThreads

PROGLIBS= RavlAudioFile.opt

EXTERNALLIBS=PulseAudioSimple.def

