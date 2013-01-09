// This file is part of OmniSoft, Pattern recognition software 
// Copyright (C) 2002, Omniperception Ltd.
// file-header-ends-here
//! docentry="Ravl.Applications.Image.Face"
//! file="Ravl/Contrib/Face/Xml/generateIdentificationProtocol.cc"

#include "Ravl/Option.hh"
#include "Ravl/EntryPnt.hh"
#include "Ravl/Stream.hh"
#include "Ravl/Stream.hh"
#include "Ravl/IO.hh"
#include "Ravl/StringList.hh"
#include "Ravl/Text/TextFile.hh"
#include "Ravl/StringList.hh"
#include "Ravl/DLIter.hh"
#include "Ravl/OS/Filename.hh"
#include "Ravl/Face/FaceInfoDb.hh"
#include "Ravl/OS/Directory.hh"
#include "Ravl/Face/FaceInfoDb.hh"
#include "Ravl/Face/EnrolSession.hh"
#include "Ravl/Face/ClaimSession.hh"
#include "Ravl/Random.hh"

using namespace RavlN::FaceN;

#define DODEBUG 0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

//! userlevel=User
//: Generates the FaceInfoDb from a list of images and the pos files 

int generateProtocolXml(int argc, char **argv)
{
  OptionC opt(argc, argv);
  FilenameC FaceInfoDbFile = opt.String("db", "/vol/db/ace/ace.xml", "the input XML file");
  DirectoryC OutDir = opt.String("o", "", "The output directory to put the test files (default same as db file)");
  UIntT noEnrolImages = opt.Int("enrol", 1, "number of enrolment images");
  IntT noTestImages = opt.Int("test", -1, "number of images to use for test");
  StringC ProtocolName = opt.String("name", "ver", "protocol name");
  bool AddProtocol = opt.Boolean("add", false, "add protocol to xml file");
  bool SingleImageClaim = opt.Boolean("single", false, "single image claim");
  UIntT noClients = opt.Int("clients", 0, "select a random number of clients");
  opt.Compulsory("name");
  opt.Check();

  //: Load in the database
  FaceInfoDbC db;
  if (!Load(FaceInfoDbFile, db))
    RavlIssueError("Trouble loading xml file");
  
  if (!opt.IsOnCommandLine("o")) {
    OutDir = FaceInfoDbFile.PathComponent();
  }
  
  //: Get sorted list
  RCHashC<StringC, DListC<FaceInfoC> > sort = db.Sort(true);
  cout << "Original number of clients: " << sort.Size() << endl;
  
  //: do we want to reduce the list
  if (opt.IsOnCommandLine("clients") && noClients < sort.Size()) {
    RCHashC<StringC, DListC<FaceInfoC> > subset;
    HashIterC<StringC, DListC<FaceInfoC> > hsh(sort);
    for (UIntT i = 0; i < noClients; i++) {
      cerr << " - chosen: " << hsh.Key() << " images: " << hsh.Data().Size() << endl;
      subset.Insert(hsh.Key(), hsh.Data());
      hsh++;
    }
    sort = subset;
    cerr << " - clients reduced to " << sort.Size() << endl;
  }

  //: Ok now we can start
  EnrolSessionC enrolSession;
  ClaimSessionC testClaimSession;
  UIntT claimNumber = 0;

  for (HashIterC<StringC, DListC<FaceInfoC> > it(sort); it; it++) {
    //: Build the xml file for the models from the first 100 people
    UIntT nImages = it.Data().Size();
    if (nImages >= noEnrolImages) {
      EnrolC enrol(it.Key());
      for (UIntT i = 0; i < noEnrolImages; i++) {
        enrol.AddFaceId(it.Data().Nth(i).FaceId());
      }
      enrolSession.Insert(it.Key(), enrol);
    } else {
      cerr << " --- Less images than required: " << it.Key() << " " << it.Data().Size() << endl;
    }
  }
  
  UIntT uptoClaimImageNo = noEnrolImages + noTestImages;
  
  //: Need to generate the claims for the evaluation
  for (HashIterC<StringC, DListC<FaceInfoC> > it(sort); it; it++) {
    UIntT nImages = sort[it.Key()].Size();
    if (nImages >= uptoClaimImageNo) {
      if (SingleImageClaim) {
        for (UIntT j = noEnrolImages; j < uptoClaimImageNo; j++) {
          ClaimC testClaim(it.Key(), it.Key(), claimNumber);
          testClaim.AddFaceId(sort[it.Key()].Nth(j).FaceId());
          testClaimSession.InsLast(testClaim);
        }
      } else {
        //: Get the face ids for the evaluation session
        DListC<StringC> testFaceIds;
        for (UIntT j = noEnrolImages; j < uptoClaimImageNo; j++) {
          testFaceIds.InsLast(sort[it.Key()].Nth(j).FaceId());
        }
        ClaimC testClaim(it.Key(), it.Key(), claimNumber);
        testClaim.AddFaceIds(testFaceIds);
        testClaimSession.InsLast(testClaim);
      }
      claimNumber++;
    } else {
      ONDEBUG(cerr << "not enough images; " << actualIt.Key() << " " << sort[actualIt.Key()].Size() << endl);
    }
  }
  cerr << "Number of searches: " << testClaimSession.Size() << endl;
  
  StringC EnrolFile = OutDir + "/" + ProtocolName + "_enrol.xml";
  StringC TestFile = OutDir + "/" + ProtocolName + "_test.xml";
  
  if (!Save(EnrolFile, enrolSession))
    RavlIssueError("Unable to save enrolSession xml file");
  
  if (!Save(TestFile, testClaimSession))
    RavlIssueError("Unable to save claimSession xml file");

  //: Check whether protocol already exists

  if (AddProtocol) {
    ProtocolC protocol(ProtocolName);
    if (db.Protocol(ProtocolName, protocol)) {
      RavlIssueWarning("Protocol with name already exists. Generate XML files but not added to db\n");
    } else {
      protocol.Type() = "identification";
      protocol.ModelFile() = EnrolFile;
      protocol.EnrolFile().InsLast(EnrolFile);
      protocol.TestFile().InsLast(TestFile);
      db.AddProtocol(protocol);
      if (!Save(FaceInfoDbFile, db))
        RavlIssueError("trouble saving database");
    }
  }

  return 0;
}

//: This puts a wrapper around the main program that catches
//: exceptions and turns them into readable error messages.

RAVL_ENTRY_POINT(generateProtocolXml);
