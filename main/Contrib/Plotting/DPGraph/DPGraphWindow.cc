// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
/////////////////////////////////////////////////
//! rcsid="$Id$"
//! lib=RavlDPGraph
//! file="Ravl/GUI/DPDisplay/DPWindow.cc"

#include "Ravl/Plot/DPGraphWindow.hh"
#include "Ravl/GUI/Manager.hh"
#include "Ravl/GUI/WaitForExit.hh"
#include "Ravl/Threads/Signal1.hh"
#include <stdlib.h>

#define DODEBUG 1
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlPlotN {
  
  DPGraphWindowBodyC::DPGraphWindowBodyC(const StringC &nname,const IndexRange2dC &nWinSize)
    : name(nname),
      winSize(nWinSize)
  {}
  
  //: Called when window is closed.
  
  bool DPGraphWindowBodyC::WindowClosed() {
    ONDEBUG(cerr << "DPGraphWindowBodyC::WindowClosed(), Called. \n");
    MutexLockC hold(accessMutex);
    //view.Invalidate();
    win.Invalidate();
    return true;
  }
  
  //: Construct widgets.
  // Access mutex must be held when calling Init().
  
  void DPGraphWindowBodyC::Init(const GuppiGraphC &initGraph) {
    ONDEBUG(cerr << "DPGraphWindowBodyC::Init(), Called.  \n");
    WaitForGUIExit(); // Make sure GUI has been started, and stop program exiting before its shutdown.
    InitGuppi();
    
    // Create a window and display it.
    win = WindowC(winSize.LCol().V()+10,winSize.BRow().V()+10,name);
    canvas = GuppiCanvasC(72*11, 72*8.5);
    graph = initGraph;
    canvas.Group().AddFull(graph);  
    win.Add(canvas);
    
    //ConnectRef(win.Signal("delete_event"),*this,&DPGraphWindowBodyC::WindowClosed);
    win.Show();
    
    // Don't setup view until we're ready to start processing data.
    //view = nview;
    ONDEBUG(cerr << "DPGraphWindowBodyC::Init(), Done.  \n");
  }
  
  //: Update data.
  
  void DPGraphWindowBodyC::Update(const StringC &name,const Array1dC<RealT> &data) {
    ONDEBUG(cerr << "DPGraphWindowBodyC::Update(), Called.  \n");
    MutexLockC hold(accessMutex);
    GuppiLineGraphC &plot = plots[name];
    if(plot.IsValid())
      plot.Update(data);
    else {
      WaitForGUIExit(); 
      InitGuppi();
      plot = GuppiLineGraphC(data);
      if(!win.IsValid()) {
	GuppiGraphC xgraph(plot);
	Init(xgraph);
      } else
	graph.AddPlot(plot);
    }
    ONDEBUG(cerr << "DPGraphWindowBodyC::Update(), Done.  \n");
  }
  
  //: Clear the display list.
  
  bool DPGraphWindowBodyC::Clear() {  
    ONDEBUG(cerr << "DPGraphWindowBodyC::Clear(), Called \n");
    MutexLockC hold(accessMutex);
#if 0
    if(!view.IsValid())
      return true;
    return view.Clear();
#else
    return true;
#endif
  }
  
  //: Refresh the display.
  
  bool DPGraphWindowBodyC::Refresh() {
    ONDEBUG(cerr << "DPGraphWindowBodyC::Refresh(), Called \n");
    MutexLockC hold(accessMutex);
#if 0
    if(!view.IsValid())
      return true;
    return view.Refresh();
#else
    return true;
#endif
  }

}
