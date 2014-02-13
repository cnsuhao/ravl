# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2001, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
###############################################
#! rcsid="$Id$"
#! file="Ravl/Applications/AVPlay/defs.mk"

DESCRIPTION = AVPlay - Audio / Video Player.

DONOT_SUPPORT=arm

PACKAGE = Ravl/VPlay

MAINS= AVPlay.cc 

PROGLIBS =  RavlImgIOV4L.opt RavlExtImgIO.opt RavlDPDisplay \
 RavlImageIO RavlVideoIO  CSPDriver.opt RavlURLIO.opt RavlLibMPEG2.opt \
 RavlImageProc RavlNet RavlDVDRead.opt RavlAVIFile.opt \
 RavlImgIOJasper.opt RavlDV.opt  RavlImgIOV4L2.opt \
 RavlRawVidIO.opt RavlLibFFmpeg.opt RavlImgIO1394dc.opt \
 RavlAV RavlVPlay RavlAudioFile.opt RavlDevAudio.opt RavlAudioIO GTK


REQUIRES = libGTK2

