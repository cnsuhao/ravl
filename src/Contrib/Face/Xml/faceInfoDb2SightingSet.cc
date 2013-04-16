// This file is part of OmniSoft, Pattern recognition software 
// Copyright (C) 2003, Omniperception Ltd.
// file-header-ends-here
// This file is part of OmniSoft, Pattern recognition software 
// Copyright (C) 2002, Omniperception Ltd.
//! rcsid="$Id: faceInfoDb2ClaimSession.cc 5236 2005-01-28 11:11:46Z kieron $"
//! docentry="Ravl.Applications.Image.Face"
//! lib=RavlFace
//! file="Ravl/Contrib/Face/Xml/modifyXml.cc"

#include "Ravl/Option.hh"
#include "Ravl/IO.hh"
#include "Ravl/RLog.hh"
#include "Ravl/Face/FaceInfoDb.hh"
#include "Ravl/Face/SightingSet.hh"

using namespace FaceN;

//! userlevel=User
//: Allows you to modify XML files

int main(int argc, char **argv) {
  OptionC opt(argc, argv);
  StringC logFile = opt.String("l", "stderr", "Checkpoint log file. ");
  StringC logLevel = opt.String("ll", "info", "Logging level (debug, info, warning, error)");
  DListC<StringC> FaceInfoDbFiles = opt.List("db", "input FaceInfoDb files");
  StringC inSightingSetFile = opt.String("i",
      "",
      "If an input claim session File is used then append new sightings to this");
  StringC SightingSetFile = opt.String("o", "sightingSet.xml", "output sighting set file");
  bool Verbose = opt.Boolean("v", false, "Do things verbosely");
  opt.Check();

  RLogInit(argc, argv, logFile.chars(), Verbose);
  RLogSubscribeL(logLevel.chars());

  SightingSetC sightingSet;
  if (opt.IsOnCommandLine("i")) {
    rInfo("Loading sighting set from '%s'", StringOf(inSightingSetFile).data());
    if (!Load(inSightingSetFile, sightingSet)) {
      rError("Unable to load sighting set file");
      return 1;
    }
  }

  FaceInfoDbC db(FaceInfoDbFiles);
  rInfo("Appending the new claims....");
  for (HashIterC<StringC, DListC<FaceInfoC> > it(db.Sort(true)); it; it++) {
    for (DLIterC<FaceInfoC> faceIt(it.Data()); faceIt; faceIt++) {
      SightingC sighting(faceIt.Data().ActualId());
      sighting.AddFaceId(faceIt.Data().FaceId());
      sightingSet.Append(sighting);
    }
  }

  rInfo("Total sightings %s", StringOf(sightingSet.Size()).data());
  rInfo("Saving to '%s'", StringOf(SightingSetFile).data());
  if (!Save(SightingSetFile, sightingSet)) {
    rError("trouble saving sighting set");
    return 1;
  }
  return 0;
}

