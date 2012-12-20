#ifndef MARK_ELEM_HH
#define MARK_ELEM_HH

//! docentry="Ravl.Applications.Video.Markup"

#include "Ravl/GUI/RadioButton.hh"
#include "Ravl/GUI/Label.hh"
#include "Ravl/GUI/Canvas.hh"
#include "Ravl/GUI/MouseEvent.hh"
#include "Ravl/GUI/Frame.hh"
#include "Ravl/GUI/Table.hh"
#include "Ravl/OS/Filename.hh"

#include <map>

using namespace RavlN;
using namespace RavlGUIN;
using namespace RavlImageN;

//! userlevel=Develop
struct MarkC
{
  int x, y, type;
};

//! userlevel=Normal
//: Manipulates the extra widgets needed for marking up
class MarkupElemC
{
public:
  MarkupElemC(CanvasC &MainCanvas, bool Verbosity, const FilenameC &TagFile="");

  MarkupElemC(MarkupElemC& me)
  {
    printf("copy constr!!!\n");
  }

  MarkupElemC(const MarkupElemC& me)
  {
    printf("const copy constr!!!\n");
  }

  MarkupElemC& operator=(MarkupElemC& me)
  {
    printf("operator=!!!\n");
    return *this;
  }
  MarkupElemC& operator=(const MarkupElemC& me)
  {
    printf("const operator=!!!\n");
    return *this;
  }

  //interface elements
  WidgetC& GetMousePosXWidget()   { return posX;         }
  WidgetC& GetMousePosYWidget()   { return posY;         }
  WidgetC& GetFrameNumberWidget() { return frameNo;      }
  WidgetC& GetControlPannel()     { return controlPannel; }
  WidgetC& GetPrevMarkWidget()    { return prevMark;     }
  WidgetC& GetNextMarkWidget()    { return nextMark;     }

  bool CallbackFrameChange(int FrameNo);
  //: Call back for frame change event

  bool CallbackSave(StringC& FileName){Save(FileName); return true;}
  //: Call back for mouse press events.

  void Load(const StringC& FileName);
  //: Load markups from file
  
  void Save(const StringC& FileName);
  //: Save markups to file

protected:
  bool CallbackMouseMotion(MouseEventC &MouseEvent);
  //: Call back for mouse movements in the window.
  
  bool CallbackMousePress(MouseEventC &MouseEvent);
  //: Call back for mouse press events.

//  bool CallbackTypeCB(bool State, int Type);
  bool CallbackTypeCB(int Type);
  //: Call back for type change

private:
  StringC fileN;

  void SetMarkType(int Type);
  void ShowMark(const MarkC& Mark);

  SArray1dC<CheckButtonC> typeBtns;
  TableC controlPannel;
  LabelC posX, posY;         //position widgets
  LabelC frameNo;            //frame number widget
  CanvasC *canvas;           //main canvas for image
  LabelC prevMark, nextMark; //widgets for frame no of prev and next marks
  CheckButtonC allowMulti;   //check button for allowing multy markers
  
  int curFrameNo;            //currentFrame number
  
  int curMarkType;
  
  bool verbosity;

  std::multimap<int, MarkC> marks;
};

#endif
