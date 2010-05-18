// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2010, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
////////////////////////////////////////////////////////////////
//! file = "Ravl/GUI/GTK/exFileChooser.cc"
//! lib = RavlGUI
//! author = "Warren Moore"

#include "Ravl/GUI/Menu.hh"
#include "Ravl/GUI/FileChooser.hh"
#include "Ravl/GUI/Manager.hh"
#include "Ravl/GUI/LBox.hh"
#include "Ravl/GUI/Window.hh"
#include "Ravl/GUI/TextBox.hh"

using namespace RavlN;
using namespace RavlGUIN;

bool FileChooserResponse(StringC &filename, TextBoxC &textBox)
{
  StringC message;
  message.form("filename(%s)\n", filename.chars());
  textBox.Insert(message);
  cerr << "FileChooserResponse " << message;
  return true;
}

bool DoQuit() 
{
  Manager.Quit();
  return true;
}

int main(int argc, char *argv[])
{
  Manager.Init(argc, argv);
  
  // Create a textbox to show files chosen.
  
  TextBoxC textBox("", true);
  
  // Create the file chooser.
  
  FileChooserC fileChooser = FileChooserC(FCA_Open, "Test Chooser", "");
  Connect(fileChooser.SigSelected(), &FileChooserResponse, "", textBox);

  DListC<StringC> allList;
  allList.InsLast("*");
  fileChooser.GUIAddFilter("All files (*)", allList);

  DListC<StringC> headerList;
  headerList.InsLast("*.hh");
  fileChooser.GUIAddFilter("Header files (*.hh)", headerList);

  DListC<StringC> imageList;
  imageList.InsLast("*.jpg");
  imageList.InsLast("*.png");
  imageList.InsLast("*.bmp");
  fileChooser.GUIAddFilter("Image files (*.jpg, *.png, *.bmp)", imageList);
  fileChooser.GUISetFilter("Image files (*.jpg, *.png, *.bmp)", imageList);

  // Create the menubar.
  
  MenuBarC menuBar(MenuC("File", 
                         MenuItemShow("Open", fileChooser) +
                         MenuItem("Quit", DoQuit)));
  
  // Create a window with a menu bar and textbox.
  
  WindowC win(100, 100, "FileChooserC Example");
  win.Add(VBox(menuBar + textBox));
  win.Show();
  
  // Start the GUI.
  
  Manager.Start();
  
  return 0;
}
