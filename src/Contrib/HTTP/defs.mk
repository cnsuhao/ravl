# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2001-11, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
#
# file-header-ends-here

REQUIRES=libCurl

#DONOT_SUPPORT=VCPP

PACKAGE=Ravl/IO

EXAMPLES = exURL.cc

HEADERS= URLStream.hh Curl.hh

SOURCES= URLStream.cc Curl.cc

MUSTLINK= URLIO.cc

PLIB= RavlURLIO

SUMMARY_LIB=Ravl

USESLIBS=RavlCore RavlIO RavlOS RavlThreads libcurl

EXTERNALLIBS= libcurl.def

EHT=Ravl.API.Core.IO.URL_Handling.html
