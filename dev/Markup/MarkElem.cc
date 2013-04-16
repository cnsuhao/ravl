#include "AVTools/MarkElem.hh"

#include "Ravl/CDLIter.hh"
#include "Ravl/OS/Date.hh"
#include "Ravl/String.hh"
#include "Ravl/GUI/LBox.hh"
#include "Ravl/DLIter.hh"

MarkupElemC::MarkupElemC(CanvasC &MainCanvas, bool Verbosity, const FilenameC &TagFile)
  : curFrameNo(0)
{
  verbosity = Verbosity;
  
  allowMulti = CheckButtonC("Allow multi");
  
  DListC<WidgetC> btns;
  if (!TagFile.IsEmpty()) {  //if file exists, read in list of button captions
    if (!TagFile.IsReadable()) {
      cerr << "cannot read class file \"" << TagFile << "\"\n";
      exit(-1);
    }
    IStreamC tagStream(TagFile);
    StringC tag;
    for(IntT i(0); readline(tagStream, tag); i++) {
      StringC label;
      label.form("%02d: %s", i, tag.chars());
      btns += CheckButtonC(label);
    }
  }
  else {
    const char* btnNames[] = { "00: SNL    ", "01: SNR    ", 
                               "02: SFL    ", "03: SFR    ", 
                               "04: HN     ", "05: HF     ", 
                               "06: BIN    ", "07: BON    ", 
                               "08: BIF    ", "09: BOF    ", 
                               "10: BINSL  ", "11: BINSR  ", 
                               "12: BONS   ", "13: BIFSL  ", 
                               "14: BIFSR  ", "15: BOFS   ", 
                               "16: NET    ", "17: TimeOut"  };
  
    const int btnNum = 18;
    for(IntT i(0); i < btnNum; i++) {
      btns += CheckButtonC(btnNames[i]);
    }  
  }

  typeBtns = SArray1dC<CheckButtonC>(btns.Size());
  DLIterC<WidgetC> p(btns);
    for(UIntT i = 0; i < btns.Size(); i++, p++)
  {
    ConnectRef(p->Signal("toggled"), *this, &MarkupElemC::CallbackTypeCB, i);    
    typeBtns[i] = *p;
  }  
  
  TableC tbl(1, 2);
  
  tbl.GUIAddObject(allowMulti, 0,1,0,1, (GtkAttachOptions) (GTK_FILL | GTK_EXPAND),GTK_FILL);
  tbl.GUIAddObject(VBox(btns), 0,1,1,2, (GtkAttachOptions) (GTK_FILL | GTK_EXPAND),GTK_FILL);

  controlPannel = tbl;

  posX = LabelC("X: --");
  posY = LabelC("Y: --");
  frameNo = LabelC("--");
  prevMark = LabelC("--");
  nextMark = LabelC("--");

  canvas = &MainCanvas;

  ConnectRef(canvas->Signal("motion_notify_event"),*this,&MarkupElemC::CallbackMouseMotion);    
  ConnectRef(canvas->Signal("button_press_event"),*this,&MarkupElemC::CallbackMousePress); 
  
  curMarkType = -1;
}

//: Call back for mouse movements in the window.
bool MarkupElemC::CallbackMouseMotion(MouseEventC &MouseEvent)
{
  Index2dC idx (MouseEvent.At()) ;  // gets the position of the mose event in RAVL co-ordinates not GTK
  posX.Label(StringC("X: ") + StringC(idx[0]));
  posY.Label(StringC("Y: ") + StringC(idx[1]));
  return true;
}

//: Call back for mouse press events.
bool MarkupElemC::CallbackMousePress(MouseEventC &MouseEvent)
{
  Index2dC idx (MouseEvent.At()) ;  // gets the position of the mouse event in RAVL co-ordinates not GTK
  if(verbosity)
    cout << "CallbackMousePress(), Called:" << curFrameNo << " : " 
         << idx[0] << " : " << idx[1] << " : " << curMarkType << endl;
  
  if(curMarkType < 0)
    return true;
         
  MarkC mark;
  mark.x = idx[0].V();
  mark.y = idx[1].V();
  mark.type = curMarkType;

  if(allowMulti.GUIIsActive())
  {
    bool found = false;
    for(multimap<int, MarkC>::iterator it = marks.lower_bound(curFrameNo); 
        it != marks.upper_bound(curFrameNo); ++it)
    {
      if(it->second.type == mark.type)
      {
        it->second = mark;
        found = true;
        break;
      }
    }
    if(!found)
    {
      marks.insert(make_pair(curFrameNo, mark));
    }
  }
  else
  {
    bool found = false;
    for(multimap<int, MarkC>::iterator it = marks.lower_bound(curFrameNo); 
        it != marks.upper_bound(curFrameNo);)
    {
      if(it->second.type == mark.type)
      {
        it->second = mark;
        found = true;
        ++it;
      }
      else
      {
        marks.erase(it++);
      }
    }
    if(!found)
    {
      marks.insert(make_pair(curFrameNo, mark));
    }
  }

  ShowMark(mark);

  return true;
}

//: Callback for frame change.
bool MarkupElemC::CallbackFrameChange(int FrameNo)
{
  if(curFrameNo != FrameNo)
  { 
    if(verbosity)
      cout << "CallbackFrameChange(), Called:" << FrameNo << endl;
    curFrameNo = FrameNo;
    frameNo.Label(StringC(FrameNo));

    multimap<int, MarkC>::iterator loIt = marks.lower_bound(curFrameNo); 
    multimap<int, MarkC>::iterator upIt = marks.upper_bound(curFrameNo);
    
    //count marks
    int numMarks = 0;
    for(multimap<int, MarkC>::iterator it = loIt; it != upIt; ++it)
      numMarks++;
    
    if(verbosity)  
      cout << "numMarks:" << numMarks << endl;
        
    //set allow multi check box  
    allowMulti.SetToggle(numMarks > 1);
    
    //clear check boxes
    for(int i = 0; i < (int)typeBtns.Size(); i++)
    {
      typeBtns[i].SetToggle(false);      
    }    
    
    //show marks
    for(multimap<int, MarkC>::iterator it = loIt; it != upIt; ++it)
      ShowMark(it->second);
    
    //show farameNo for previous mark
    if(loIt != marks.begin())
    {
      loIt--;
    }    
    if(loIt != marks.end() && curFrameNo > loIt->first)
    {
      prevMark.Label(StringC(loIt->first));
    }
    else
    {
      prevMark.Label(StringC("--"));
    }

    //show frameNo for next mark
    upIt = marks.upper_bound(curFrameNo);
    if(upIt != marks.end() && curFrameNo < upIt->first)
    {
      nextMark.Label(StringC(upIt->first));
    }
    else
    {
      nextMark.Label(StringC("--"));    
    }
  }

  return true;
}

//: Call back for type change
bool MarkupElemC::CallbackTypeCB(int Type)
{
  bool state = typeBtns[Type].IsActive();
//  cout << "Type:" << Type << " \tState:" << state << endl;
  if(state)
    SetMarkType(Type);
  else //delete mark
  {
    for(multimap<int, MarkC>::iterator it = marks.lower_bound(curFrameNo); 
        it != marks.upper_bound(curFrameNo); ++it)
    {
      if(it->second.type == Type)
      {
        if(verbosity)
          cout << "Deleting mark: " << curFrameNo << ":" << Type << endl;
        marks.erase(it);
        break;
      }
    }
  
    curMarkType = -1;
  }
    
  return true;
}

void MarkupElemC::SetMarkType(int Type)
{
  if(verbosity)
    cout << "Set mark type:" << Type << endl;
  curMarkType = Type;
  if(allowMulti.IsActive())
  {    
    typeBtns[Type].SetToggle(true);
  }
  else
  {
    for(int i = 0; i < (int)typeBtns.Size(); i++)
    {
      if(i == Type)
      {
        //if(!typeBtns[i].IsActive())
          typeBtns[i].SetToggle(true);
      }
      else
      {
        //if(typeBtns[i].IsActive())
          typeBtns[i].SetToggle(false);      
      }
    }
  }
}

void MarkupElemC::ShowMark(const MarkC& Mark)
{
  if(verbosity)
    cout << "Show mark:" << Mark.x << ":" << Mark.y << ":" << Mark.type << endl;
  Sleep(0.3);//wait for other thread to draw image on canvas not well implemented FIXME (AK)
  SetMarkType(Mark.type);
  canvas->DrawLine(Mark.y - 3, Mark.x - 3, Mark.y + 3, Mark.x + 3, 1);
  canvas->DrawLine(Mark.y - 3, Mark.x + 3, Mark.y + 3, Mark.x - 3, 1);
}

void MarkupElemC::Load(const StringC& FileName)
{
  fileN = FileName;

  marks.clear();
  IStreamC inFile(FileName);
  int counter = 0;
  while(1)
  {
    MarkC mark;
    int fn = -1;

    inFile >> fn;
    inFile >> mark.x;
    inFile >> mark.y;
    inFile >> mark.type;

    if(fn >= 0)
    {
      marks.insert(make_pair(fn, mark));
      counter++;
    }
    else
    {
      break;
    }  
  };

  if(verbosity)
    printf("Read %i elements from file %s\n", counter, FileName.chars());
}

void MarkupElemC::Save(const StringC& FileName)
{
  if(FileName.IsEmpty())
  {
    printf("Name of file is not specified\n");
    return;
  }

  int counter = 0;

  if(verbosity)
    printf("We must write %i elements\n", int(marks.size()));

  OStreamC outFile(FileName);
  for(std::multimap<int, MarkC>::iterator curMark = marks.begin();
      curMark != marks.end(); curMark++)
  {
    outFile << curMark->first << '\t' << curMark->second.x << '\t'
            << curMark->second.y << '\t' << curMark->second.type << endl;
    if(verbosity)
      cout    << curMark->first << '\t' << curMark->second.x << '\t'
              << curMark->second.y << '\t' << curMark->second.type << endl;
    counter++;
  }
  if(verbosity)
    printf("Written %i elements to file %s\n", counter, FileName.chars());
}


