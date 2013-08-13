# This file is part of OmniSoft, Pattern recognition software 
# Copyright (C) 2002, Omniperception Ltd.
# file-header-ends-here
#! file="Contrib/Face/Experiment/defs.mk"

LICENSE=Copyright

REQUIRES=RLog

ORGANISATION=

PACKAGENAME=RavlFace

DESCRIPTION=Specification of face data sets using XML

PACKAGE = Ravl/Face

MAINS= doClassifierRoc.cc

SOURCES = Roc.cc

HEADERS = Roc.hh

MUSTLINK=linkFaceExperiment.cc

PLIB = RavlFace

LIBDEPS=RavlFaceRoc.def

USESLIBS= RavlFace

PROGLIBS = RavlExtImgIO.opt RavlPatternRecIO RavlPatternRec

SCRIPTS=

EHT=
