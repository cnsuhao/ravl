// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
/////////////////////////////////////////////////////////
//! rcsid="$Id$"
//! lib=RavlGUI
//! file="Ravl/GUI/GTK/exList.cc"

#include "Ravl/GUI/Window.hh"
#include "Ravl/GUI/Manager.hh"
#include "Ravl/GUI/List.hh"

using namespace RavlGUIN;

int main(int nargs,char *args[]) 
{
  Manager.Init(nargs,args);
  
  WindowC win(100,100,"Hello");
  using namespace RavlGUIN;
  DListC<Tuple2C<IntT,StringC> > sel;
  sel.InsFirst(Tuple2C<IntT,StringC>(1,"Hello1"));
  sel.InsFirst(Tuple2C<IntT,StringC>(2,"Hello2"));
  sel.InsFirst(Tuple2C<IntT,StringC>(3,"Hello3"));
  sel.InsFirst(Tuple2C<IntT,StringC>(4,"A longer message"));
  //MenuBarC menuBar;
  //menuBar +
  ListC aList(sel);
  aList.AppendLine(6,StringC("Hi!!!"));
  win.Add(aList);
  aList.AppendLine(5,StringC("Hi2"));
  
  win.Show();
  aList.RemoveLine(2);
  Manager.Start();
  
}
