# Copyright (C) 2002, Omniperception Ltd.
# file-header-ends-here
#! file="Contrib/Face/Experiment/defs.mk"

LICENSE=Copyright

REQUIRES=RLog

ORGANISATION=

DESCRIPTION=Specification of face data sets using XML

PACKAGE = Ravl/Face

MAINS= doClassifierRoc.cc

SOURCES = Roc.cc

HEADERS = Roc.hh

MUSTLINK=linkFaceExperiment.cc

PLIB = RavlFace

SUMMARY_LIB=Ravl

LIBDEPS=RavlFaceRoc.def

USESLIBS= RavlFace

PROGLIBS = RavlExtImgIO.opt RavlPatternRecIO RavlPatternRec

SCRIPTS=

EHT=
