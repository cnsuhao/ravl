#ifndef RAVLGUI_DRAGANDDROP_HEADER
#define RAVLGUI_DRAGANDDROP_HEADER 1
//! author="Charles Galambos"
//! date="15/06/2002"
//! userlevel=Normal
//! docentry="Ravl.GUI.Control"

#include "Ravl/GUI/Widget.hh"

namespace RavlGUIN {
  
  //! userlevel=Normal
  //: Position and time of an event.
  
  class PositionTimeC {
  public:
    PositionTimeC()
    {}
    //: Default constructor.
    
    PositionTimeC(const Index2dC &nat,UIntT ntime)
      : at(nat),
	time(ntime)
    {}
    //: Constructor.

    PositionTimeC(int x,int y,UIntT ntime)
      : at(x,y),
	time(ntime)
    {}
    //: Constructor.
    
    Index2dC &At()
    { return at; }
    //: Access location.
    
    UIntT Time() const
    { return time; }
    //: Access time of click
    
    Index2dC at; // Position of click.
    UIntT time;  // Time of click.
  };
  
  //! userlevel=Normal
  //: Drag and drop info.
  
  class DNDDataInfoC {
  public:
    DNDDataInfoC()
      : context(0),
	data(0),
	info(0),
	time(0)
    {}
    //: Default constructor.
    
    DNDDataInfoC(GdkDragContext *dc,GtkSelectionData *data,UIntT ninfo,UIntT ntime,int x = 0 ,int y = 0)
      : context(dc),
	data(data),
	info(ninfo),
	time(ntime),
	at(x,y)
    {}
    //: Default constructor.
    
    bool Finish(bool success,bool del);
    //: Finish drag and drop.
    // This must be called on the GUI thread.
    
    bool IsString() const;
    //: Is recieved data a string ?
    
    StringC String();
    //: Get data as string.
    // It is the user's responsibility to ensure this is appropriate.
    
    GtkWidget *GTKSourceWidget();
    // Find the GTK source widget.
    
    GtkSelectionData *Data()
    { return data; }
    //: Access data.
    
    UIntT Info() const
    { return info; }
    //: Info field.
    
    Index2dC &At()
    { return at; }
    //: Return location of event.
    
    UIntT Time() const
    { return time; }
    //: Return time of event.
    
  protected:
    // Info....
    GdkDragContext *context;
    GtkSelectionData *data;
    UIntT info;
    UIntT time;  // Time of click.
    Index2dC at;
  };

  
}


#endif
