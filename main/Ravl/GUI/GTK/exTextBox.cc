// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
/////////////////////////////////////////////////////////
//! rcsid="$Id$"
//! lib=RavlGUI
//! file="Ravl/GUI/GTK/exTextBox.cc"
//! author="Charles Galambos"
//! date="23/03/1999"
//! docentry="Ravl.API.GUI.Control"

#include "Ravl/GUI/Window.hh"
#include "Ravl/GUI/TextBox.hh"
#include "Ravl/GUI/TextEntry.hh"
#include "Ravl/GUI/Manager.hh"
#include "Ravl/GUI/LBox.hh"
#include "Ravl/Option.hh"

using namespace RavlGUIN;

int main(int nargs,char *args[]) 
{
  Manager.Init(nargs,args);
  OptionC opts(nargs,args);
  opts.Check();
  
  WindowC win(100,100,"Hello");
  TextBoxC textBox("TextBoxC");
  TextEntryC entry("TextEntryC");
  TextEntryC pw("Password");
  bool bTrue = true;
  pw.HideText(bTrue);
  win.Add(VBox(entry+pw+textBox));
  win.Show();  
  Manager.Start();
}
