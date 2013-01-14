// This file is part of OmniSoft, Pattern recognition software 
// Copyright (C) 2003, Omniperception Ltd.
// file-header-ends-here
//! rcsid="$Id$"
//! docentry="Ravl.Applications.Image.Face"
//! lib=RavlFace
//! file="Ravl/Contrib/Face/splitFaceXml.cc"

#include "Ravl/Option.hh"
#include "Ravl/OS/Filename.hh"
#include "Ravl/EntryPnt.hh"
#include "Ravl/Stream.hh"
#include "Ravl/IO.hh"
#include "Ravl/Face/FaceInfoDb.hh"
#include "Ravl/Face/EnrolSession.hh"
#include "Ravl/Face/ClaimSession.hh"
#include "Ravl/StringList.hh"
#include "Ravl/Text/TextFile.hh"
#include "Ravl/DLIter.hh"

using namespace RavlN::FaceN;

//! userlevel=User
//: Allows you to modify XML files

int splitXml(int argc, char **argv) {
  OptionC opt(argc, argv);
  DListC<StringC> FaceInfoDbFiles = opt.List("db", "input FaceInfoDb files");
  StringC outFile1 = opt.String("o1", "out1.xml", "output file 1");
  StringC outFile2 = opt.String("o2", "out2.xml", "output file 2");
  UIntT n1 = opt.Int("n1", 3, "number of faces per subject first file");
  UIntT n2 = opt.Int("n2", 3, "number of faces per subject in second file");
  UIntT maxSubjects = opt.Int("subjects", 0, "Maximum number of clients");

  //: check options selection
  opt.Check();

  //: Lets modify any of the FaceInfoDb files
  //=================================
  FaceInfoDbC faceInfoDb(FaceInfoDbFiles);

  FaceInfoDbC xml1;
  FaceInfoDbC xml2;
  UIntT subjects = 0;
  for (HashIterC<StringC, DListC<FaceInfoC> >it(faceInfoDb.Sort(true)); it; it++) {

    if(it.Data().Size() < (n1 + n2)) {
      continue;
    }

    UIntT c = 0;
    for (DLIterC<FaceInfoC>it2(it.Data()); it2; it2++) {
      if (c < n1) {
        xml1.Insert(it2.Data().FaceId(), it2.Data());
      } else if (c < (n1 + n2)) {
        xml2.Insert(it2.Data().FaceId(), it2.Data());
      }
      c++;
    }

    subjects++;
    // check subject limit not reached
    if(maxSubjects != 0 && subjects >= maxSubjects)
      break;
  }

  //: Lets save some
  Save(outFile1, xml1);
  Save(outFile2, xml2);


  return 0;
}

//: This puts a wrapper around the main program that catches
//: exceptions and turns them into readable error messages.


RAVL_ENTRY_POINT(splitXml);
