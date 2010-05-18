// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2010, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
////////////////////////////////////////////////////////////////
//! file = "Ravl/GUI/GTK/FileChooser.hh"
//! lib = RavlGUI
//! author = "Warren Moore"
//! docentry = "Ravl.API.Graphics.GTK.Control"
//! example = "exFileChooser.cc"

#ifndef RAVLGUI_FILECHOOSER_HEADER
#define RAVLGUI_FILECHOOSER_HEADER 1

#include "Ravl/GUI/Widget.hh"
#include "Ravl/String.hh"
#include "Ravl/Threads/Signal1.hh"
#include "Ravl/Threads/Signal3.hh"
#include <gtk/gtk.h>

namespace RavlGUIN
{

  enum FileChooserActionT
  {
    FCA_Open, // Select a file to open.
    FCA_Save, // Select a file to save.
    FCA_SelectFolder, // Select a folder.
    FCA_CreateFolder // Select or create a folder.
  };

  class FileChooserC;

  //! userlevel=Develop
  //: File chooser.
  
  class FileChooserBodyC
  : public WidgetBodyC
  {
  public:
    FileChooserBodyC(const FileChooserActionT action,
                     const StringC &title,
                     const StringC &filename,
                     const bool confirmOverwrite,
                     const bool hideOnResponse,
                     const bool sendEmptyStringOnCancel);
    //: Ctor.
    //!param: action - The dialog mode: open file, save file, open folder or save to folder.
    //!param: title - The dialog title.
    //!param: filename - When opening, select the specified file or folder. When saving, suggest the specified file or folder.
    //!param: confirmOverwrite - Enable or disable the default overwrite confirmation dialog.
    //!param: hideOnResponse - If enabled, the dialog will disappear when one of the dialog buttons is clicked.
    //!param: sendEmptyStringOnCancel - If enabled, a signal containing an empty string is emitted if cancel is clicked or the dialog is closed.

    virtual bool Create();

    bool GUISetTitle(const StringC &title);
    //: Set the dialog title.

    bool SetTitle(const StringC &title);
    //: Set the dialog title.

    bool GUISetFilename(const StringC &filename);
    //: When opening, select the specified file or folder. When saving, suggest the specified file or folder.
    // NOTE: This will reset any added or set file filters (as of GTK 2.18.6).

    bool SetFilename(const StringC &filename);
    //: When opening, select the specified file or folder. When saving, suggest the specified file or folder.
    // NOTE: This will reset any added or set file filters (as of GTK 2.18.6).

    bool GUISetConfirmOverwrite(const bool confirmOverwrite);
    //: Enable or disable the default overwrite confirmation dialog.

    bool SetConfirmOverwrite(const bool confirmOverwrite);
    //: Enable or disable the default overwrite confirmation dialog.

    void SetHideOnResponse(const bool hideOnResponse)
    { m_hideOnResponse = hideOnResponse; }
    //: If enabled, the dialog will disappear when one of the dialog buttons is clicked.

    bool HideOnResponse() const
    { return m_hideOnResponse; }
    //: Will the dialog will disappear when one of the dialog buttons is clicked?

    void SetSendEmptyStringOnCancel(const bool sendEmptyStringOnCancel)
    { m_sendEmptyStringOnCancel = sendEmptyStringOnCancel; }
    //: If enabled, a signal containing an empty string is emitted if cancel is clicked or the dialog is closed.

    bool SendEmptyStringOnCancel() const
    { return m_sendEmptyStringOnCancel; }
    //: Will a signal containing an empty string be emitted if cancel is clicked or the dialog is closed?

    bool GUISetFilter(const StringC &name, const DListC<StringC> patterns);
    //: Set the currently selected filter.
    // NOTE: This filter does not need to be separately added with AddFilter().
    //!param: name - The name of the filter e.g. "Image files".
    //!param: patterns - A list of shell globs e.g. a list containing the strings "*.jpg", "*.png" and "*.bmp".

    bool SetFilter(const StringC &name, const DListC<StringC> patterns);
    //: Set the currently selected filter.
    // NOTE: This filter does not need to be separately added with AddFilter().
    //!param: name - The name of the filter e.g. "Image files".
    //!param: patterns - A list of shell globs e.g. a list containing the strings "*.jpg", "*.png" and "*.bmp".

    bool GUIAddFilter(const StringC &name, const DListC<StringC> patterns);
    //: Add a user selectable filter.
    //!param: name - The name of the filter e.g. "Image files".
    //!param: patterns - A list of shell globs e.g. a list containing the strings "*.jpg", "*.png" and "*.bmp".

    bool AddFilter(const StringC &name, const DListC<StringC> patterns);
    //: Add a user selectable filter.
    //!param: name - The name of the filter e.g. "Image files".
    //!param: patterns - A list of shell globs e.g. a list containing the strings "*.jpg", "*.png" and "*.bmp".

    Signal1C<StringC> &SigSelected()
    { return m_sigSelected; }
    //: Signal emitted when a file or folder is selected. If SendEmptyStringOnCancel is enabled, an empty string is emitted if cancel is clicked or the dialog is closed.

  protected:
    bool DoSetFilename();

    bool DoSetFilter();

    bool DoAddFilter(const StringC &name, const DListC<StringC> patterns);
    
    virtual void Destroy();
    
    StringC m_title;
    FileChooserActionT m_action;
    StringC m_filename;
    bool m_confirmOverwrite;
    bool m_hideOnResponse;
    bool m_sendEmptyStringOnCancel;
    StringC m_filterSetName;
    DListC<StringC> m_filterSetPatterns;
    DListC<Tuple2C<StringC, DListC<StringC> > > m_filterList;

    Signal1C<StringC> m_sigSelected;
    
    friend class FileChooserC;
  };
  
  //! userlevel=Normal
  //: File chooser.
  
  class FileChooserC
  : public WidgetC
  {
  public:
    FileChooserC(bool)
    : WidgetC(*new FileChooserBodyC(FCA_Open, "File Chooser", "", true, true, true))
    {}
    //: Ctor.

    FileChooserC(const FileChooserActionT action = FCA_Open,
                 const StringC &title = "File Chooser",
                 const StringC &filename = "",
                 const bool confirmOverwrite = true,
                 const bool hideOnResponse = true,
                 const bool sendEmptyStringOnCancel = true)
    : WidgetC(*new FileChooserBodyC(action, title, filename, confirmOverwrite, hideOnResponse, sendEmptyStringOnCancel))
    {}
    //: Ctor.
    //!param: action - The dialog mode: open file, save file, open folder or save to folder.
    //!param: title - The dialog title.
    //!param: filename - When opening, select the specified file or folder. When saving, suggest the specified file or folder.
    //!param: confirmOverwrite - Enable or disable the default overwrite confirmation dialog.
    //!param: hideOnResponse - If enabled, the dialog will disappear when one of the dialog buttons is clicked.
    //!param: sendEmptyStringOnCancel - If enabled, a signal containing an empty string is emitted if cancel is clicked or the dialog is closed.
    
    FileChooserC(FileChooserBodyC &body)
    : WidgetC(body)
    {}
    //: Body ctor.

    bool GUISetTitle(const StringC &title)
    { return Body().GUISetTitle(title); }
    //: Set the dialog title.

    bool SetTitle(const StringC &title)
    { return Body().SetTitle(title); }
    //: Set the dialog title.

    bool GUISetFilename(const StringC &filename)
    { return Body().GUISetFilename(filename); }
    //: When opening, select the specified file or folder. When saving, suggest the specified file or folder.
    // NOTE: This will reset any added or set file filters (as of GTK 2.18.6).

    bool SetFilename(const StringC &filename)
    { return Body().SetFilename(filename); }
    //: When opening, select the specified file or folder. When saving, suggest the specified file or folder.
    // NOTE: This will reset any added or set file filters (as of GTK 2.18.6).

    bool GUISetConfirmOverwrite(const bool confirmOverwrite)
    { return Body().GUISetConfirmOverwrite(confirmOverwrite); }
    //: Enable or disable the default overwrite confirmation dialog.

    bool SetConfirmOverwrite(const bool confirmOverwrite)
    { return Body().SetConfirmOverwrite(confirmOverwrite); }
    //: Enable or disable the default overwrite confirmation dialog.

    void SetHideOnResponse(const bool hideOnResponse)
    { Body().SetHideOnResponse(hideOnResponse); }
    //: If enabled, the dialog will disappear when one of the dialog buttons is clicked.

    bool HideOnResponse() const
    { return Body().HideOnResponse(); }
    //: Will the dialog will disappear when one of the dialog buttons is clicked?

    void SetSendEmptyStringOnCancel(const bool sendEmptyStringOnCancel)
    { Body().SetSendEmptyStringOnCancel(sendEmptyStringOnCancel); }
    //: If enabled, a signal containing an empty string is emitted if cancel is clicked or the dialog is closed.

    bool SendEmptyStringOnCancel() const
    { return Body().SendEmptyStringOnCancel(); }
    //: Will a signal containing an empty string be emitted if cancel is clicked or the dialog is closed?

    bool GUISetFilter(const StringC &name, const DListC<StringC> patterns)
    { return Body().GUISetFilter(name, patterns); }
    //: Set the currently selected filter.
    // NOTE: This filter does not need to be separately added with AddFilter().
    //!param: name - The name of the filter e.g. "Image files".
    //!param: patterns - A list of shell globs e.g. a list containing the strings "*.jpg", "*.png" and "*.bmp".

    bool SetFilter(const StringC &name, const DListC<StringC> patterns)
    { return Body().SetFilter(name, patterns); }
    //: Set the currently selected filter.
    // NOTE: This filter does not need to be separately added with AddFilter().
    //!param: name - The name of the filter e.g. "Image files".
    //!param: patterns - A list of shell globs e.g. a list containing the strings "*.jpg", "*.png" and "*.bmp".

    bool GUIAddFilter(const StringC &name, const DListC<StringC> patterns)
    { return Body().GUIAddFilter(name, patterns); }
    //: Add a user selectable filter.
    //!param: name - The name of the filter e.g. "Image files".
    //!param: patterns - A list of shell globs e.g. a list containing the strings "*.jpg", "*.png" and "*.bmp".

    bool AddFilter(const StringC &name, const DListC<StringC> patterns)
    { return Body().AddFilter(name, patterns); }
    //: Add a user selectable filter.
    //!param: name - The name of the filter e.g. "Image files".
    //!param: patterns - A list of shell globs e.g. a list containing the strings "*.jpg", "*.png" and "*.bmp".

    Signal1C<StringC> &SigSelected()
    { return Body().SigSelected(); }
    //: Signal emitted when a file or folder is selected. If SendEmptyStringOnCancel is enabled, an empty string is emitted if cancel is clicked or the dialog is closed.

  protected:
    FileChooserBodyC &Body()
    { return static_cast<FileChooserBodyC&>(WidgetC::Body()); }
    //: Body access.
    
    const FileChooserBodyC &Body() const
    { return static_cast<const FileChooserBodyC&>(WidgetC::Body()); }
    //: Body access.
  };
  
}

#endif
