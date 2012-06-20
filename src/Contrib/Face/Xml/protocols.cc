// This file is part of OmniSoft, Pattern recognition software 
// Copyright (C) 2003, Omniperception Ltd.
// file-header-ends-here
// This file is part of OmniSoft, Pattern recognition software 
// Copyright (C) 2002, Omniperception Ltd.
//! docentry="Ravl.Contrib.Face"
//! lib=RavlFace
//! file="Ravl/Contrib/Face/Xml/modifyXml.cc"

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

using namespace RavlN::FaceN;

//! userlevel=User
//: Look at the protocols in an XML file

int main(int argc, char **argv)
{
  OptionC opt(argc, argv);
  DListC<StringC> FaceInfoDbFiles = opt.List("db", "input FaceInfoDb files");
  bool listProtocols = opt.Boolean("list", false, "list all protocols");
  StringC protocolName = opt.String("name", "", "protocol name");
  bool listModelFile = opt.Boolean("model", false, "list model files for chosen protocol");
  bool listEnrolFile = opt.Boolean("train", false, "list enrol files for chosen protocol");
  bool listEvalFile = opt.Boolean("eval", false, "list evaluation files for chosen protocol");
  bool listTestFile = opt.Boolean("test", false, "list test files for chosen protocol");
  //: check options selection
  opt.Check();
  
  FaceInfoDbC faceInfoDb(FaceInfoDbFiles);
  
  DListC<ProtocolC> protocols = faceInfoDb.Protocols();
  
  if (listProtocols) {
    for (DLIterC<ProtocolC> it(protocols); it; it++)
      cout << it.Data().Name() << endl;
  }

  if (opt.IsOnCommandLine("name")) {

    //: Check chosen protocol exists
    ProtocolC protocol;
    for (DLIterC<ProtocolC> it(protocols); it; it++) {
      if (it.Data().Name() == protocolName)
        protocol = *it;
    }
    if (!protocol.IsValid()) {
      cout << "did not find protocol" << endl;
      exit(0);
    }

    //: OK lets see what we have been asked to do
    if (listModelFile) {
      cout << protocol.ModelFile() << endl;
    }
    
    if (listEnrolFile) {
      for (DLIterC<FilenameC> it(protocol.EnrolFile()); it; it++)
        cout << *it << endl;
    }

    if (listEvalFile) {
      for (DLIterC<FilenameC> it(protocol.EvalFile()); it; it++)
        cout << *it << endl;
    }

    if (listTestFile) {
      for (DLIterC<FilenameC> it(protocol.TestFile()); it; it++)
        cout << *it << endl;
    }
    
  }

  return 0;
}

