// This file is part of OmniSoft, Pattern recognition software
// Copyright (C) 2003, Omniperception Ltd.
// file-header-ends-here
//! file="OmniSoft/Applications/WhoFIT/ControlWin.cc"
//! lib=OmniWhoFIT

#include "ControlWin.hh"

#include "Ravl/GUI/Manager.hh"
#include "Ravl/GUI/FileSelector.hh"
#include "Ravl/GUI/Label.hh"
#include "Ravl/GUI/MessageBox.hh"
#include "Ravl/GUI/PackInfo.hh"
#include "Ravl/RLog.hh"

namespace RavlN {
  namespace FaceN {

    using namespace RavlGUIN;

    bool gui_quit()
    {
      Manager.Quit(); // Initate shutdown.
      return true;
    }

    //: Constructor.
    ControlWinBodyC::ControlWinBodyC(DListC<StringC> & dbNames, bool autoScale) :
        WindowBodyC(800, 600, "Face XML Markup Tool"), notebook(GTK_POS_TOP, true, true), faceDb(dbNames), m_autoScale(autoScale)
    {
      if (faceDb.IsEmpty()) {
        rWarning("No markups in file provided!");
        gui_quit();
      }
    }

    bool ControlWinBodyC::Create()
    {

      Add(notebook);

      //: view page 0
      viewPage = ViewPageC(faceDb, m_autoScale);
      pages.InsLast(viewPage);
      notebook.AppendPage(viewPage, LabelC("Database"));

      //: show default page
      notebook.ShowPage(pages.First());

      // Create window
      return WindowBodyC::Create();;
    }

    bool ControlWinBodyC::Quit()
    {
      // Quit the GUI
      Manager.Quit();
      return true;
    }

    bool ControlWinBodyC::Refresh()
    {
      return true;
    }

  }
}
