# This file is part of OmniSoft, Pattern recognition software 
# Copyright (C) 2002, Omniperception Ltd.
# file-header-ends-here
#! rcsid="$Id$"
#! file="Contrib/Face/Xml/defs.mk"

LICENSE=Copyright

REQUIRES=RLog

ORGANISATION=OmniPerception Ltd

PACKAGENAME=RavlFace

DESCRIPTION=Specification of face data sets using XML

PACKAGE =  Ravl/Face


MAINS= readFaceXml.cc copyFaceXml.cc modifyFaceXml.cc splitFaceXml.cc \
 faceInfoDb2enrolSession.cc protocols.cc insertFeatureSets.cc mergeSightingSets.cc

SOURCES =   FaceInfo.cc Enrol.cc EnrolSession.cc Expert.cc Score.cc \
 ScoreSession.cc Protocol.cc FaceInfoDb.cc FaceInfoStream.cc Sighting.cc \
 SightingSet.cc FacePair.cc FacePairSet.cc Claim.cc ClaimSession.cc \
 ScoresTable.cc ResultsInfo.cc Experiment.cc

HEADERS =  FaceInfo.hh Enrol.hh EnrolSession.hh Expert.hh Score.hh \
 ScoreSession.hh Protocol.hh FaceInfoDb.hh FaceInfoStream.hh Sighting.hh \
 SightingSet.hh FacePair.hh FacePairSet.hh Claim.hh ClaimSession.hh \
 ScoresTable.hh ResultsInfo.hh Experiment.hh

PLIB    = RavlFace

LIBDEPS= 

USESLIBS=RavlMath RavlImageProc RavlCore RavlIO RavlImage RavlOS \
 RavlXMLFactory RavlRLog RavlGnuPlot RavlDataSet

PROGLIBS = RavlExtImgIO.opt  RavlPatternRecIO

SCRIPTS=

EHT=
