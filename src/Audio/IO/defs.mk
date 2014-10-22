# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2003-14, OmniPerception Ltd.
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
#! file="Ravl/Audio/IO/defs.mk"

PACKAGE=Ravl/Audio

HEADERS=SphereIO.hh SphereFormat.hh TranscriptionFiles.hh \
 TranscriptionStream.hh PlaySound.hh

SOURCES=SphereIO.cc SphereFormat.cc TranscriptionFiles.cc \
 TranscriptionStream.cc PlaySound.cc AudioSample.cc

PLIB=RavlAudioIO

SUMMARY_LIB=Ravl

USESLIBS=RavlAudioUtil RavlOS RavlOSIO

MUSTLINK=RavlAudioSample.cc

MAINS=audioconv.cc

PROGLIBS=RavlOSIO RavlDevAudio.opt RavlAudioFile.opt RavlPortAudio.opt

EXAMPLES= exTranscriptionStream.cc
