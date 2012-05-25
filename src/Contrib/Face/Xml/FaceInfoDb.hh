// This file is part of OmniSoft, Pattern recognition software
// Copyright (C) 2003, Omniperception Ltd.
// file-header-ends-here
#ifndef RAVLN_FACEN_FACEINFODB_HEADER
#define RAVLN_FACEN_FACEINFODB_HEADER 1
//! rcsid="$Id$"
//! lib=RavlFace
//! docentry="Ravl.Contrib.Face"
//! file="Ravl/Contrib/Face/Xml/FaceInfoDb.hh"
//! author="Kieron J Messer"

#include "Ravl/RefCounter.hh"
#include "Ravl/XMLStream.hh"
#include "Ravl/RCHash.hh"
#include "Ravl/Face/FaceInfo.hh"
#include "Ravl/Face/EnrolSession.hh"
#include "Ravl/Face/Protocol.hh"
#include "Ravl/OS/Directory.hh"
#include "Ravl/XMLFactoryRegister.hh"

namespace RavlN {
  namespace FaceN {

    //! userlevel="Normal"
    //: A RCHashC of StringC and FaceInfoC with XML streams
    //: The string is the unique id of face, taken to be the leading

    class FaceInfoDbC : public RCHashC<StringC, FaceInfoC>
    {

    public:

      FaceInfoDbC();
      //: Constructor

      FaceInfoDbC(const StringC & dbName);
      //: Construct from a known database

      FaceInfoDbC(DListC<StringC> & dbName);
      //: Construct from a known database

      FaceInfoDbC(const XMLFactoryContextC &node);
      //: Construct from an XML factory

    protected:

      FaceInfoDbC(XMLIStreamC& xml);
      //: Constructor from XML stream

      XMLOStreamC& SaveXML(XMLOStreamC &xml) const;
      //: Save to xml stream

    public:

      static FaceInfoDbC GenerateXml(const DirectoryC & dir, const StringC & splitChar = "_", UIntT n = 0);
      //: Generate XML from images in a directory.

      static FaceInfoDbC GenerateXml(const DListC<DirectoryC> & dirs, const StringC & splitChar = "_", UIntT n = 0);
      //: Generate XML from images in a set of directories

      static FaceInfoDbC GenerateXml(const DListC<StringC> & imageFiles, const StringC & name);
      //: Generate XML for a single subject from a set of image files

      UIntT NoFaces() const
      {
        return Size();
      }
      //: The number of faces

      UIntT NoClients() const;
      //: The number of individual clients

      DListC<StringC> Keys(bool MarkedUp = false) const;
      //: A list of all the faceids in the database, this is sorted alphabetically

      SArray1dC<StringC> AllFaces() const;
      //: An array of all the face ids in the database.

      DListC<StringC> Clients() const;
      //: A list of all the clients in the database

      RCHashC<StringC, DListC<FaceInfoC> > Sort(bool markedUp = false) const;
      //: Sort into client ids and then faceids

      FaceInfoDbC Search(const StringC & tag, const StringC & value, bool exactMatch = true) const;
      //: Return sub-set of database (search on tags "pose", "spectrum", "glasses", "id", "normalisation")

      FaceInfoDbC Search(const StringC & tag, const DListC<StringC> & values) const;
      //: Return sub-set of database (search on tags "pose", "spectrum", "glasses", "id", "normalisation")

      DListC<StringC> NormImages(const StringC & type) const;
      //: Get a list of all the norm images for a type

      bool ModifyImagePath(const StringC & from, const StringC & to);
      //: Modifies the original image path

      bool DeleteId(const StringC & id);
      //: delete all images associated with an id

      bool RenameId(const StringC & from, const StringC & to, bool actualIdOnly = false);
      //: Rename id

      bool KeepId(const EnrolSessionC & enrolSession);
      //: Only keep the ids that appear in the enrol session

      FaceInfoDbC Ids(const EnrolSessionC & enrolSession) const;
      //: Return new db with only these face ids

      StringC& Name()
      {
        return m_strName;
      }
      //: Get database name

      void Name(const StringC & strName)
      {
        m_strName = strName;
      }
      //: Set database name

      StringC Root() const
      {
        return m_root;
      }
      //: Get database name

      void Root(const StringC & root)
      {
        m_root = root;
      }
      //: Set database name

      DListC<ProtocolC> Protocols(const StringC & type = "") const;
      //: Protocols

      bool Protocol(const StringC & name, ProtocolC & protocol) const;
      //: Get a protocol of a given name

      void Protocols(DListC<ProtocolC> & prots)
      {
        protocols = prots;
      }
      //: Protocols

      void AddProtocol(const ProtocolC & protocol)
      {
        protocols.InsLast(protocol);
      }

    protected:
      void init(DListC<StringC> & dbNames);

      bool ExpandKnownDatabases2(DListC<StringC> & databases);
      StringC m_root;
      //: Root of database (optional)

      StringC m_strName;
      //: Database name (optional)

      DListC<ProtocolC> protocols;
      //: has the database got some protocols

      friend XMLIStreamC &operator>>(XMLIStreamC &xml, FaceInfoDbC &data);
      friend XMLOStreamC &operator<<(XMLOStreamC &xml, const FaceInfoDbC &data);

    };

    XMLIStreamC &operator>>(XMLIStreamC &xml, FaceInfoDbC &data);
    //: XML Input
    // DO NOT USE DIRECTLY unless you're using this inside another XML loader and you know what you're doing.
    // Use Load() instead.

    XMLOStreamC &operator<<(XMLOStreamC &xml, const FaceInfoDbC &data);
  //: XML Output
  // DO NOT USE DIRECTLY unless you're using this inside another XML output func and you know what you're doing.
  // Use Save() instead.

  }
}

#endif
