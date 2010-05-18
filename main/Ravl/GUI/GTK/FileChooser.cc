// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2010, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
////////////////////////////////////////////////////////////////
//! file = "Ravl/GUI/GTK/FileChooser.cc"
//! lib = RavlGUI
//! author = "Warren Moore"

#include "Ravl/GUI/FileChooser.hh"
#include "Ravl/OS/Filename.hh"
#include "Ravl/GUI/Manager.hh"

#define DODEBUG 0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlGUIN
{

  static void CBFileChooserResponse(GtkWidget *widget, gint response, FileChooserBodyC *fileChooserBodyPtr)
  {
    ONDEBUG(cerr << "CBFileChooserResponse response(" << response << ")" << endl);
    RavlAssert(fileChooserBodyPtr);

    if (fileChooserBodyPtr->HideOnResponse())
      fileChooserBodyPtr->GUIHide();

    bool fileSelected = false;
    StringC filename;
    switch (response)
    {
      case GTK_RESPONSE_OK:
      case GTK_RESPONSE_ACCEPT:
      {
        char *filenamePtr = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(widget));
        if (filenamePtr)
        {
          fileSelected = true;
          filename = filenamePtr;
        }
        break;
      }
      case GTK_RESPONSE_CANCEL:
      case GTK_RESPONSE_DELETE_EVENT:
      default:
        break;
    }

    if (fileSelected || fileChooserBodyPtr->SendEmptyStringOnCancel())
      fileChooserBodyPtr->SigSelected()(filename);
  }

  FileChooserBodyC::FileChooserBodyC(const FileChooserActionT action,
                                     const StringC &title,
                                     const StringC &filename,
                                     const bool confirmOverwrite,
                                     const bool hideOnResponse,
                                     const bool sendEmptyStringOnCancel)
  : m_title(title),
    m_action(action),
    m_filename(filename),
    m_confirmOverwrite(confirmOverwrite),
    m_hideOnResponse(hideOnResponse),
    m_sendEmptyStringOnCancel(sendEmptyStringOnCancel),
    m_sigSelected(StringC(""))
  {}
  
  bool FileChooserBodyC::Create()
  {
    ONDEBUG(cerr << "FileChooserBodyC::Create" << endl);
    
    GtkFileChooserAction action = GTK_FILE_CHOOSER_ACTION_OPEN;
    StringC actionButton = GTK_STOCK_OPEN;

    switch (m_action)
    {
      case FCA_Save:
        action = GTK_FILE_CHOOSER_ACTION_SAVE;
        actionButton = GTK_STOCK_SAVE;
        break;
      case FCA_SelectFolder:
        action = GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER;
        break;
      case FCA_CreateFolder:
        action = GTK_FILE_CHOOSER_ACTION_CREATE_FOLDER;
        actionButton = GTK_STOCK_SAVE;
        break;
      case FCA_Open:
      default:
        break;
    }

    widget = gtk_file_chooser_dialog_new(m_title,
                                         NULL,
                                         action,
                                         GTK_STOCK_CANCEL, GTK_RESPONSE_CANCEL,
                                         actionButton.chars(), GTK_RESPONSE_ACCEPT,
                                         NULL);

    gtk_file_chooser_set_local_only(GTK_FILE_CHOOSER(widget), true);

    GUISetConfirmOverwrite(m_confirmOverwrite);

    if (m_filename != "")
      DoSetFilename();

    if (m_filterList.Size() > 0)
      for (DLIterC<Tuple2C<StringC, DListC<StringC> > > filterIter(m_filterList); filterIter; filterIter++)
        DoAddFilter(filterIter->Data1(), filterIter->Data2());

    if (m_filterSetPatterns.Size() > 0)
      DoSetFilter();

    ConnectSignals(); 
    
    gtk_signal_connect(GTK_OBJECT(GTK_DIALOG(widget)),
                       "response",
                       (GtkSignalFunc)CBFileChooserResponse,
                       this);

    return true;
  }

  bool FileChooserBodyC::GUISetTitle(const StringC &title)
  {
    m_title = title;
    
    if (widget)
    {
      RavlAssertMsg(Manager.IsGUIThread(), "Incorrect thread. This method may only be called on the GUI thread.");

      gtk_window_set_title(GTK_WINDOW(widget), m_title);
    }
    
    return true;
  }

  bool FileChooserBodyC::SetTitle(const StringC &title)
  {
    Manager.Queue(Trigger(FileChooserC(*this), &FileChooserC::GUISetTitle, title));
    return true;
  }

  bool FileChooserBodyC::GUISetFilename(const StringC &filename)
  {
    m_filename = filename;

    if (widget)
    {
      return DoSetFilename();
    }

    return true;
  }

  bool FileChooserBodyC::SetFilename(const StringC &filename)
  {
    Manager.Queue(Trigger(FileChooserC(*this), &FileChooserC::GUISetFilename, filename));
    return true;
  }

  bool FileChooserBodyC::GUISetConfirmOverwrite(const bool confirmOverwrite)
  {
    m_confirmOverwrite = confirmOverwrite;
    
    if (widget)
    {
      RavlAssertMsg(Manager.IsGUIThread(), "Incorrect thread. This method may only be called on the GUI thread.");

      gtk_file_chooser_set_do_overwrite_confirmation(GTK_FILE_CHOOSER(widget), confirmOverwrite);
    }

    return true;
  }

  bool FileChooserBodyC::SetConfirmOverwrite(const bool confirmOverwrite)
  {
    Manager.Queue(Trigger(FileChooserC(*this), &FileChooserC::GUISetConfirmOverwrite, confirmOverwrite));
    return true;
  }

  bool FileChooserBodyC::GUISetFilter(const StringC &name, const DListC<StringC> patterns)
  {
    m_filterSetName = name;
    m_filterSetPatterns = patterns;
    
    if (widget)
    {
      return DoSetFilter();
    }

    return true;
  }

  bool FileChooserBodyC::SetFilter(const StringC &name, const DListC<StringC> patterns)
  {
    Manager.Queue(Trigger(FileChooserC(*this), &FileChooserC::GUISetFilter, name, patterns));
    return true;
  }

  bool FileChooserBodyC::GUIAddFilter(const StringC &name, const DListC<StringC> patterns)
  {
    if (patterns.Size() == 0)
      return true;

    m_filterList.InsLast(Tuple2C<StringC, DListC<StringC> >(name, patterns));

    if (widget)
    {
      GtkFileFilter *filter = gtk_file_chooser_get_filter(GTK_FILE_CHOOSER(widget));
      if (filter)
      {
        const char *filterName = gtk_file_filter_get_name(filter);
        if (filterName && name == filterName)
        {
          gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(widget), filter);
          return true;
        }
      }
      
      return DoAddFilter(name, patterns);
    }

    return true;
  }

  bool FileChooserBodyC::AddFilter(const StringC &name, const DListC<StringC> patterns)
  {
    Manager.Queue(Trigger(FileChooserC(*this), &FileChooserC::GUIAddFilter, name, patterns));
    return true;
  }

  bool FileChooserBodyC::DoSetFilename()
  {
    RavlAssertMsg(Manager.IsGUIThread(), "Incorrect thread. This method may only be called on the GUI thread.");
    RavlAssert(widget);

    FilenameC filenameObject(m_filename);
    if (filenameObject.Exists())
    {
      gtk_file_chooser_set_filename(GTK_FILE_CHOOSER(widget), filenameObject);
    }
    else
    {
      gtk_file_chooser_set_current_folder(GTK_FILE_CHOOSER(widget), filenameObject.PathComponent());
      if (m_action == FCA_Save || m_action == FCA_CreateFolder)
        gtk_file_chooser_set_current_name(GTK_FILE_CHOOSER(widget), filenameObject.NameComponent());
    }
    
    return true;
  }

  bool FileChooserBodyC::DoSetFilter()
  {
    RavlAssertMsg(Manager.IsGUIThread(), "Incorrect thread. This method may only be called on the GUI thread.");
    RavlAssert(widget);

    if (m_filterList.Size() > 0)
    {
      GSList *listedFilters = gtk_file_chooser_list_filters(GTK_FILE_CHOOSER(widget));
      if (listedFilters)
      {
        GSList *listedFilterIter = listedFilters;
        while (listedFilterIter)
        {
          GtkFileFilter *filter = GTK_FILE_FILTER(listedFilterIter->data);
          const char *filterName = gtk_file_filter_get_name(filter);
          ONDEBUG(cerr << "FileChooserBodyC::GUISetFilter name(" << m_filterSetName << ") checking name(" << filterName << ")" << endl);
          if (filterName && m_filterSetName == filterName)
          {
            ONDEBUG(cerr << "FileChooserBodyC::GUISetFilter name(" << m_filterSetName << ") found name(" << filterName << ")" << endl);
            gtk_file_chooser_set_filter(GTK_FILE_CHOOSER(widget), filter);
            break;
          }

          listedFilterIter = listedFilterIter->next;
        }

        g_slist_free(listedFilters);
      }
    }
    else
    {
      GtkFileFilter *filter = gtk_file_filter_new();
      gtk_file_filter_set_name(filter, m_filterSetName);
      for (DLIterC<StringC> patternIter(m_filterSetPatterns); patternIter; patternIter++)
      {
        ONDEBUG(cerr << "FileChooserBodyC::GUISetFilter name(" << m_filterSetName << ") adding pattern(" << *patternIter << ")" << endl);
        gtk_file_filter_add_pattern(filter, *patternIter);
      }

      gtk_file_chooser_set_filter(GTK_FILE_CHOOSER(widget), filter);
    }

    return true;
  }

  bool FileChooserBodyC::DoAddFilter(const StringC &name, const DListC<StringC> patterns)
  {
    RavlAssertMsg(Manager.IsGUIThread(), "Incorrect thread. This method may only be called on the GUI thread.");
    RavlAssert(widget);

    GtkFileFilter *filter = gtk_file_filter_new();
    gtk_file_filter_set_name(filter, name);
    for (DLIterC<StringC> patternIter(patterns); patternIter; patternIter++)
    {
      ONDEBUG(cerr << "FileChooserBodyC::GUIAddFilter name(" << name << ") adding pattern(" << *patternIter << ")" << endl);
      gtk_file_filter_add_pattern(filter, *patternIter);
    }

    gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(widget), filter);

    return true;
  }

  void FileChooserBodyC::Destroy()
  {
    m_sigSelected.DisconnectAll();
    WidgetBodyC::Destroy();
  }

}
