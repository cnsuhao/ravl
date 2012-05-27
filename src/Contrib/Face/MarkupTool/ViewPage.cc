// This file is part of OmniSoft, Pattern recognition software
// Copyright (C) 2003, Omniperception Ltd.
// file-header-ends-here
//! file="OmniSoft/Applications/WhoFIT/MainWindow.cc"
//! lib=OmniWhoFIT

#include "ViewPage.hh"

#include "Ravl/IO.hh"

#include "Ravl/GUI/MessageBox.hh"
#include "Ravl/GUI/MarkupImageRGB.hh"
#include "Ravl/GUI/Frame.hh"
#include "Ravl/GUI/Label.hh"
#include "Ravl/GUI/ScrolledArea.hh"
#include "Ravl/GUI/PackInfo.hh"
#include "Ravl/GUI/FileSelector.hh"
#include "Ravl/RLog.hh"

#include <gdk/gdkevents.h>
//#include "Ravl/GUI/Manager.hh"
//#include "Ravl/GUI/LBox.hh"

namespace RavlN {
  namespace FaceN {

    using namespace RavlGUIN;

    bool StringCmpFunc(const StringC &E1, const StringC &E2)
    {
      return E1.CaseCmp(E2) <= 0;
    }

    /////////////////////////////////////
    //: Construct from a subject table
    //: and an xml face database
    /////////////////////////////////////

    ViewPageBodyC::ViewPageBodyC(FaceInfoDbC & db, bool autoScale) :
        LBoxBodyC(true), faceDb(db), m_autoScale(autoScale), canvas(800, 800), m_markupMode(false)
    {

      StringC normDir = "unknown";
      faceDb.Root(normDir);
      //: lets get a list of all our faceids and point the iterator to the first
      faceIds = faceDb.Keys();
      iter = DLIterC<StringC>(faceIds);
      faceDbFile = db.Name();

    }

    //////////////////////////////////////
    //: Create the GUI
    //////////////////////////////////////

    bool ViewPageBodyC::Create()
    {

      //: the box that holds the path
      imagepath = TextEntryC("unknown");
      //imageSize = TextEntryC("unknown");
      //dateOfBirth = TextEntryC(DateC::NowLocal().ODBC());
      textEntryPose = TextEntryC("unknown");

      //: Previous and next buttons
      ButtonC next = ButtonR("Next", *this, &ViewPageBodyC::NextPrevButton, 1);
      ButtonC prev = ButtonR("Prev", *this, &ViewPageBodyC::NextPrevButton, 2);
      ButtonC save = ButtonR("Save", *this, &ViewPageBodyC::SaveButton);

      //: Make the tree store
      SArray1dC<AttributeTypeC> types(2);
      types[0] = AttributeTypeStringC("ID", "...");
      types[1] = AttributeTypeStringC("Face ID", "...");
      treeStore = TreeStoreC(types);
      UpdateTreeStore();
      treeView = TreeViewC(treeStore);
      treeView.SetAttribute(1, "resizable", "1", false);
      ConnectRef(treeView.SelectionChanged(), *this, &ViewPageBodyC::SelectRow);

      //: Glasses selector
      noglasses = RadioButton("Yes", glassesButtonGrp);
      glasses = RadioButton("No", glassesButtonGrp);
      unknown = RadioButton("Unknown", glassesButtonGrp);

      //: Sex selector
      //male = RadioButton("Male", sexButtonGrp);
      //female = RadioButton("Female", sexButtonGrp);
      //unknownsex = RadioButton("Unknown", sexButtonGrp);

      //: date of image
      date = TextEntryC(DateC::NowLocal().ODBC());

      FrameC glassesFrame(VBox(noglasses + glasses + unknown), "Glasses");
      FrameC poseFrame(textEntryPose, "Pose");
      //FrameC sexFrame(VBox(male + female + unknownsex), "Sex");
      //LabelC dobLabel("Date of Birth");
      //FrameC dobFrame(HBox(dobLabel + dateOfBirth), "Other Info");

      //: This is the layout
      LabelC dateLabel("Capture Date");
      Add(VBox(PackInfoC(HBox(PackInfoC(VBox(PackInfoC(canvas, true) + PackInfoC(save, false)), true)
          + PackInfoC(ScrolledAreaC(treeView, 350, 30), false)),
          true)
          + PackInfoC(HBox(LabelC("Path") + imagepath + HBox(dateLabel + date)), false)
          + PackInfoC(HBox(glassesFrame + poseFrame), false)));

      // Connect an event handler to the frame widget.
      ConnectRef(canvas.Signal("key_press_event"), *this, &ViewPageBodyC::ProcessKeyboard);

      fs = FileSelectorC("Save XML file", "*.xml");
      ConnectRef(fs.Selected(), *this, &ViewPageBodyC::Save);

      LoadData();

      // Create window
      return LBoxBodyC::Create();
    }

    //////////////////////////////////////////////////////////
    //: This is called when a new subject has been selected
    //: Any information that has been updated on the page
    //: is saved into memory
    /////////////////////////////////////////////////////////

    bool ViewPageBodyC::SaveData()
    {
      rDebug("Saving face '%s'", iter.Data().data());
      FaceInfoC info = faceDb[*iter];

      //: The eye positions might of changed
      //////////////////////////////////////
      for (HashIterC<IntT, MarkupPoint2dC> it(m_markupPoints); it; it++) {
        ImagePointFeatureC feature;
        if (!info.FeatureSet().Feature(it.Key(), feature)) {
          rWarning("Urm, could not find feature whilst trying to save!");
          continue;
        }

        if (feature.Location() != it.Data().Position()) {
          rDebug("Changed position of %s", feature.Description().data());
          info.FeatureSet().Set(it.Key(), it.Data().Position());
        } else {
          rDebug("No need to save point '%s'", feature.Description().data());
        }

      }

      return true;
    }

    //////////////////////////////////////////////
    //: Load in data in the main data viewer page
    //////////////////////////////////////////////

    bool ViewPageBodyC::LoadData()
    {

      //: Check iter is pointing at sensible stuff
      if (!faceDb.IsElm(*iter)) {
        //RavlIssueWarning("WARNING: selected image not held in database.\n");
        return false;
      }

      rDebug("Loading face '%s'", iter.Data().data());

      //: Load in the image from file
      ///////////////////////////////
      FaceInfoC info = faceDb[*iter];
      ImageC<ByteRGBValueC> im;
      if (!Load(info.OrigImage(), im)) {
        rWarning("Image unable to be loaded %s", info.OrigImage().data());
        AlertBox("Unable to load image");
        return false;
      }

      //: Add the image to the markup
      ///////////////////////////////
      FrameMarkupC fm;
      image = im.Copy();
      fm.Markup().InsLast(MarkupImageRGBC(-1, 0, image));

      if (m_markupMode) {
        if (m_markupPoints.IsEmpty()) {
          if (info.FeatureSet().IsValid()) {
            //DListC<Point2dC> points;
            for (HashIterC<IntT, ImagePointFeatureC> it(info.FeatureSet().FeatureIterator()); it; it++) {
              rDebug("Displaying point '%s'", it.Data().Description().data());
              MarkupPoint2dC mup(1, 1, it.Data().Location(), MP2DS_CrossHair);
              fm.Markup().InsLast(mup);
              m_markupPoints.Insert(it.Key(), mup);
              //points.InsLast(it.Data().Location());
            }
          }
        } else {
          for (HashIterC<IntT, MarkupPoint2dC> it(m_markupPoints); it; it++) {
            fm.Markup().InsLast(it.Data());
          }
        }
      } else {
        if (info.FeatureSet().IsValid()) {
          //DListC<Point2dC> points;
          for (HashIterC<IntT, ImagePointFeatureC> it(info.FeatureSet().FeatureIterator()); it; it++) {
            rDebug("Displaying point '%s'", it.Data().Description().data());
            MarkupPoint2dC mup(1, 1, it.Data().Location(), MP2DS_CrossHair);
            fm.Markup().InsLast(mup);
            m_markupPoints.Insert(it.Key(), mup);
            //points.InsLast(it.Data().Location());
          }
        }
      }

      //: Scale if we have eye-points..turn this off just so it remembers last scale
      ////////////////////////////////////////////
      if (m_autoScale) {
        RealT canvasRows = canvas.SizeY();
        RealT canvasCols = canvas.SizeX();
        //RealT currentScale = canvas.Scale().Row();
        //rInfo("Current Scale %0.2f", currentScale);

        RealT scale;
        Point2dC offset;
        if (image.Rows() > image.Cols()) {
          scale = canvasRows / (RealT) image.Rows();
          offset = Point2dC(0.0, 0.5 * (canvasCols - (scale * image.Cols())));

        } else {
          scale = canvasCols / (RealT) image.Cols();
          offset = Point2dC(0.5 * (canvasRows - (scale * image.Rows())), 0.0);
        }
        //rInfo("Scale: %0.2f", scale);
        canvas.SetScale(Vector2dC(scale, scale));
        canvas.SetOffset(offset);
      }

      //: Update the canvas with all our information
      //////////////////////////////////////////////
      canvas.UpdateMarkup(fm);
      //StringC imageInfo = (StringC) im.Rows() + " " + (StringC) im.Cols();
      //imageSize.Text(imageInfo);

      //: Set the image path
      imagepath.Text(faceDb[*iter].OrigImage());

      //: Check whether image has glasses
      if (info.Glasses() == "no")
        noglasses.SetToggle(true);
      else if (info.Glasses() == "yes")
        glasses.SetToggle(true);
      else
        unknown.SetToggle(true);

      //: Check the pose information
      textEntryPose.Text(info.Pose());

      //: Check the date of image
      date.Text(info.Date().ODBC());

      return true;
    }

    //////////////////////////////////////////////
    //: Will sequentially walk through images in
    //: the tree store
    ///////////////////////////////////////////////

    bool ViewPageBodyC::NextPrevButton(IntT & v)
    {
      //: work out whether glasses
      StringC desc;
      if (v == 1) {
        SaveData();
        iter++;
        if (!iter) {
          // end reached
          iter.Nth(-1);
        }
      } else if (v == 2) {
        SaveData();
        iter--;
        if (!iter) {
          // beginning reached
          iter.Nth(0);
        }
      }
      LoadData();
      return true;
    }

    //////////////////////////////////////////////

    bool ViewPageBodyC::Save(const StringC & filename)
    {
      rInfo("Saving to file '%s'", filename.data());
      if (!FaceN::Save(filename, faceDb)) {
        AlertBox(StringC("trouble saving database"));
        return false;
      }
      return true;
    }

    //////////////////////////////////////////////

    bool ViewPageBodyC::SaveButton()
    {
      fs.GUIShow();
      return true;
    }

    //////////////////////////////////////////////////////
    //: Called when a new row is selectd in the tree store
    //////////////////////////////////////////////////////

    bool ViewPageBodyC::SelectRow()
    {
      SaveData();
      StringC faceId;
      DListC<TreeModelIterC> selected = treeView.GUISelected();
      if (selected.IsEmpty())
        return false; // none selected, do not do anything
      if (!treeStore.GetValue(selected.First(), 1, faceId))
        return false; // had trouble getting value, do nothing
      for (iter.First(); iter; iter++) {
        if (iter.Data() == faceId) {
          LoadData();
          return true;
        }
      }
      return false;
    }

    bool ViewPageBodyC::ProcessKeyboard(GdkEventKey *KeyEvent)
    {
      //cerr << "Key pressed : " << KeyEvent->keyval << "\n";
      if (KeyEvent->keyval == 65364) {
        DListC<TreeModelIterC> selected = treeView.GUISelected();
        if (selected.IsEmpty())
          return true; // none selected, do not do anything
        TreeModelIterC &tvIter = selected.First();
        if (tvIter.HasChildren()) {
          m_lastParent = tvIter.Copy();
          treeView.GUIExpand(tvIter);
          treeView.GUISelectIter(tvIter.Children());

        } else if (tvIter.Next()) {
          treeView.GUISelectIter(tvIter);
        } else {
          if (m_lastParent.IsElm()) {
            if (m_lastParent.Next()) {
              treeView.GUISelectIter(m_lastParent);
            }
          }

        }
      }
      // right key
      else if (KeyEvent->keyval == 65363) {
        treeView.GUIExpandAll();
      }
      // left key
      else if (KeyEvent->keyval == 65361) {
        treeView.GUICollapseAll();
      }
      // m key - toggle markup mode
      else if (KeyEvent->keyval == 109) {
        if (m_markupMode) {
          rInfo("Turning off mark-up mode!");
          m_markupMode = false;
          m_autoScale = true;
        } else {
          rInfo("Turning on mark-up mode!");
          m_markupMode = true;
          m_autoScale = false;
        }
      }

      return true; //event processed
    }

    /////////////////////////////////////////////////
    //: Called when we need to update the information
    //: held by the tree store
    //////////////////////////////////////////////////

    bool ViewPageBodyC::UpdateTreeStore()
    {
      TreeModelIterC iter1;
      TreeModelIterC iter2;
      RCHashC<StringC, DListC<FaceInfoC> > sorted = faceDb.Sort(true); // only get marked up images
      rInfo("Number of Clients %s and Number of Images %s", StringOf(sorted.Size()).data(), StringOf(faceDb.Size()).data());
      DListC<StringC> clients = faceDb.Clients();
      for (DLIterC<StringC> it(clients); it; it++) {
        DListC<FaceInfoC> faces = sorted[*it];
        if (!faces.IsEmpty()) {
          DLIterC<FaceInfoC> faceIt(faces);
          treeStore.AppendRow(iter1);
          //: do the first row
          treeStore.GUISetValue(iter1, 0, it.Data());
          treeStore.GUISetValue(iter1, 1, faceIt.Data().FaceId());

          for (faceIt++; faceIt; faceIt++) {
            treeStore.AppendRow(iter2, iter1);
            treeStore.GUISetValue(iter2, 0, StringC(""));
            treeStore.GUISetValue(iter2, 1, faceIt.Data().FaceId());
          }

        }
      }
      return true;
    }

  }
}
