// This file is part of OmniSoft, Pattern recognition software
// Copyright (C) 2003, Omniperception Ltd.
// file-header-ends-here
#ifndef VIEWPAGE_HEADER
#define VIEWPAGE_HEADER 1
//! file="OmniSoft/Applications/WhoFIT/ViewPage.hh"
//! docentry = "Applications.Mainment Tool"
//! author   = "Kieron Messer"
//! userlevel=Develop
//! lib=OmniWhoFIT

#include "Ravl/GUI/LBox.hh"

#include "Ravl/Face/FaceInfoDb.hh"

#include "Ravl/GUI/Menu.hh"
#include "Ravl/GUI/GUIMarkupCanvas.hh"
#include "Ravl/GUI/MarkupPoint2d.hh"
#include "Ravl/GUI/RadioButton.hh"
#include "Ravl/GUI/TextEntry.hh"
#include "Ravl/GUI/TreeStore.hh"
#include "Ravl/GUI/TreeView.hh"
#include "Ravl/GUI/FileSelector.hh"
#include "Ravl/Threads/Mutex.hh"

namespace RavlN {
  namespace FaceN {

    using namespace RavlGUIN;
    
    class ViewPageBodyC : public LBoxBodyC
    {

    public:

      ViewPageBodyC(FaceInfoDbC & db, bool autoScale);
      //: Constructor.

      bool Create();
      // Creates the window

      bool SaveData();
      //: Save data

      bool LoadData();
      //: Load data

      bool NextPrevButton(IntT & v);

      bool SelectRow();
      //: Select row

      bool ProcessKeyboard(GdkEventKey *KeyEvent);
      //: process keyboard event

      bool UpdateTreeStore();
      //: Update the tree store

      bool Save(const StringC & filename);
      //: Save the xml file

      bool SaveButton();
      //: Save the xml file

      bool SaveSelectedButton();
      //: Just save the XML which have been selected

      bool SaveSelected(const StringC & filename);
      //: Just save the XML which have been selected

      bool DeleteButton();
      //: Save the xml file

    protected:
      FaceInfoDbC faceDb;
      //: A database of the faces

      bool m_autoScale;
      //: Do we want to scale the images

      DListC<StringC> faceIds;
      //: A list of all the face ids to display

      DLIterC<StringC> iter;
      //: The iterator which points at the image we are looking at

      RCHashC<StringC, bool>m_selected;

      StringC faceDbFile;
      //: The faceDb file to save as

      //: The GUI to display the marked up face
      GUIMarkupCanvasC canvas;

      RCHashC<IntT, MarkupPoint2dC> m_markupPoints;
      ImageC<ByteRGBValueC> image; // copy of the current image

      RCHashC<IntT, MarkupPoint2dC> m_prevMarkupPoints;

      TextEntryC imagepath;
      TreeViewC treeView;
      TreeStoreC treeStore;

      //: Glasses
      RadioButtonGroupT glassesButtonGrp;
      RadioButtonC noglasses;
      RadioButtonC glasses;
      RadioButtonC unknown;

      //: Pose
      TextEntryC textEntryPose;


      //: Date of image
      TextEntryC date;
      FileSelectorC m_fileSelectorSave;
      FileSelectorC m_fileSelectorSaveSelected;

      bool m_markupMode;

      MutexC m_mutex;

      TreeModelIterC m_lastParent;

      friend class ViewPageC;

    };

    //! userlevel=Basic

    class ViewPageC : public LBoxC
    {

    public:
      ViewPageC()
      {
      }
      //: Default constructor.
      // Creates an invalid handle.

      ViewPageC(FaceInfoDbC & db, bool autoScale) :
          LBoxC(*new ViewPageBodyC(db, autoScale))
      {
      }
      //: Constructor.

    protected:

      ViewPageC(ViewPageBodyC &bod) :
          LBoxC(bod)
      {
      }
      //: Body Constructor.

      ViewPageBodyC &Body()
      {
        return static_cast<ViewPageBodyC &>(LBoxC::Body());
      }
      //: Access body.

      const ViewPageBodyC &Body() const
      {
        return static_cast<const ViewPageBodyC &>(LBoxC::Body());
      }
      //: Access body.

    public:

      friend class ViewPageBodyC;

    };

  }
}
#endif //MAIN_WINDOW_HEADER
