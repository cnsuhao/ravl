# This file is part of OmniSoft, Pattern recognition software 
# Copyright (C) 2002, Omniperception Ltd.
# file-header-ends-here
#! file="Contrib/Face/Xml/defs.mk"

LICENSE=Copyright

REQUIRES=RLog

ORGANISATION=

PACKAGENAME=RavlFace

DESCRIPTION=Specification of face data sets using XML

PACKAGE = Ravl/Face

MAINS=

SOURCES = Roc.cc

HEADERS = Roc.hh

PLIB = RavlFace

LIBDEPS= 

USESLIBS= RavlMath RavlImageProc RavlCore RavlIO RavlImage RavlOS \
 RavlXMLFactory RavlRLog RavlGnuPlot

PROGLIBS = RavlExtImgIO.opt RavlPatternRec RavlPatternRecIO

SCRIPTS=

EHT=
