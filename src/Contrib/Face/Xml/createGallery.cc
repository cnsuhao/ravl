// This file is part of OmniSoft, Pattern recognition software 
// Copyright (C) 2003, Omniperception Ltd.
// file-header-ends-here
// This file is part of OmniSoft, Pattern recognition software 
// Copyright (C) 2002, Omniperception Ltd.
//! rcsid="$Id$"
//! docentry="Ravl.Applications.Image.Face"
//! lib=RavlFace
//! file="Ravl/Contrib/Face/Xml/modifyFaceXml.cc"

#include "Ravl/Option.hh"
#include "Ravl/OS/Filename.hh"
#include "Ravl/EntryPnt.hh"
#include "Ravl/Stream.hh"
#include "Ravl/IO.hh"
#include "Ravl/Face/SightingSet.hh"
#include "Ravl/Face/FaceInfoDb.hh"
#include "Ravl/StringList.hh"
#include "Ravl/Text/TextFile.hh"
#include "Ravl/Collection.hh"
#include "Ravl/CollectionIter.hh"
#include "Ravl/RLog.hh"

using namespace RavlN;
using namespace RavlN::FaceN;

//! userlevel=User
//: Create a gallery from a sighting set

int createGallery(int argc, char **argv)
{
  OptionC opt(argc, argv);
  StringListC DbFiles = opt.List("db", "The input database files");
  StringListC SightingFiles = opt.List("sighting", "", "input sighting files");
  StringC OutGalleryFile = opt.String("gallery", "newGallery.xml", "output gallery file");
  StringC OutSightingFile = opt.String("o", "newSighting.xml", "output sighting file");

  // RLog options
  bool verbose = opt.Boolean("v", false, "Verbose mode. ");
  StringC logFile = opt.String("l", "stderr", "Checkpoint log file. ");
  StringC logLevel = opt.String("ll", "debug", "Logging level (debug, info, warning, error)");

  //: check options selection
  opt.Check();

  // Set up logging
  RavlN::RLogInit(argc, argv, logFile, verbose);
  RavlN::RLogSubscribeL(logLevel.chars());

  FaceInfoDbC faceDb(DbFiles);

  SightingSetC sightingSet;
  for(DLIterC<StringC>it(SightingFiles);it;it++) {


    SightingSetC thisSightingSet;
    if(!Load(*it, thisSightingSet)) {
      rWarning("Unable to load sighting set '%s'", it.Data().data());
      continue;
    }
    rInfo("Loaded '%s' and has %s sightings", it.Data().data(), StringOf(thisSightingSet.Size()).data());
    sightingSet.Append(thisSightingSet);
  }


  SightingSetC outSightingSet;

  RCHashC<StringC, bool >alreadyDone;
  FaceInfoDbC galleryDb;
  for(DArray1dIterC<SightingC>it(sightingSet);it;it++) {

    StringC sightingActualId = it.Data().ActualId();

    if(alreadyDone.IsElm(sightingActualId)) {
      outSightingSet.Append(*it);
      continue;
    }

    for(DLIterC<StringC>faceIt(it.Data().FaceIds());faceIt;faceIt++) {

      if(!faceDb.IsElm(*faceIt)) {
        rDebug("Face not in FaceDb");
        continue;
      }

      galleryDb.Insert(*faceIt, faceDb[*faceIt]);
      alreadyDone.Insert(sightingActualId, true);
    }

  }


  if(!Save(OutGalleryFile, galleryDb)) {
    rInfo("Failed to save gallery!");
    return 1;
  }

  rInfo("Saving XML file to '%s' with %s sightings", OutSightingFile.data(), StringOf(outSightingSet.Size()).data());
  if (!Save(OutSightingFile, outSightingSet)) {
    rError("Trouble saving XML file '%s'", OutSightingFile.data());
    return 1;
  }

  return 0;
}

//: This puts a wrapper around the main program that catches
//: exceptions and turns them into readable error messages.

RAVL_ENTRY_POINT(createGallery);

