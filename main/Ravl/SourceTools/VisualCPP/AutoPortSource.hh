#ifndef RAVL_SOURCEAUTOPORT_HEADER
#define RAVL_SOURCEAUTOPORT_HEADER 1
/////////////////////////////////////////////////////////
//! rcsid="$Id$"
//! userlevel=Normal

#include "Ravl/SourceTools/DefsFile.hh"
#include "Ravl/SourceTools/LibInfo.hh"
#include "Ravl/SourceTools/ProgInfo.hh"

namespace RavlN {
  
  class AutoPortSourceC;
  
  //: Code Porter Body
  
  class AutoPortSourceBodyC 
    : public RCBodyVC
  {
  public:
    AutoPortSourceBodyC(StringC &where);
    //: Constructor.
    
    bool ScanTree(StringC &where);
    //: Scan a tree for info.
    
    bool ScanDirectory(StringC &where,DefsMkFileC &defs);
    //: Scan the contents of the directory.
    
    bool Unix2Dos(const StringC &src,const StringC &vcpp);
    //: Update a file.
    // vcpp, is the file in the Visual C++ tree.
    // src, is the unix version. <p>
    
    bool Dos2Unix(const StringC &src,const StringC &vcpp);
    //: Update a file.
    // vcpp, is the file in the Visual C++ tree.
    // src, is the unix version. <p>
    
    void SetVerbose(bool val)
      { verbose = val; }
    //: Set verbose flag.
    
    HashC<StringC,LibInfoC> &Libs()
      { return libs; }
    //: Access table of libraries.

    DListC<ProgInfoC> &Mains()
      { return mains; }
    //: Main programs in tree.
    
    DListC<ProgInfoC> &Tests()
      { return tests; }
    //: Test programs.
    
    DListC<ProgInfoC> &Examples()
      { return examples; }
    //: Examples programs.
    
  protected:
    bool verbose;
    
    HashC<StringC,LibInfoC> libs;
    DListC<ProgInfoC> mains;
    DListC<ProgInfoC> tests;
    DListC<ProgInfoC> examples;
    
    friend class AutoPortSourceC;
  };
  
  //: Code Porting tool.
  
  class AutoPortSourceC 
    : public RCHandleC<AutoPortSourceBodyC>
  {
  public:
    AutoPortSourceC()
      {}
    //: Default constructor.
    // Creates an invalid handle.
    
    AutoPortSourceC(StringC &where)
      : RCHandleC<AutoPortSourceBodyC>(*new AutoPortSourceBodyC(where))
      { Body().ScanTree(where); }
    //: Constructor.
    // scan's the source tree 'where' for info.
    
  protected:
    AutoPortSourceC(AutoPortSourceBodyC &bod)
      : RCHandleC<AutoPortSourceBodyC>(bod)
      {}
    //: Body constructor.
    
    AutoPortSourceBodyC &Body()
      { return RCHandleC<AutoPortSourceBodyC>::Body(); }
    //: Access body.
    
    const AutoPortSourceBodyC &Body() const
      { return RCHandleC<AutoPortSourceBodyC>::Body(); }
    //: Access body.
    
  public:
    bool ScanDirectory(StringC &where,DefsMkFileC &defs)
      { return Body().ScanDirectory(where,defs); }
    //: Scan the contents of the directory.

    void SetVerbose(bool val)
      { Body().SetVerbose(val); }
    //: Set verbose flag.

    HashC<StringC,LibInfoC> &Libs()
      { return Body().Libs(); }
    //: Access table of libraries.

    DListC<ProgInfoC> &Mains()
      { return Body().Mains(); }
    //: Main programs in tree.
    
    DListC<ProgInfoC> &Tests()
      { return Body().Tests(); }
    //: Test programs.
    
    DListC<ProgInfoC> &Examples()
      { return Body().Examples(); }
    //: Examples programs.
    
    friend class AutoPortSourceBodyC;
  };
  
}

#endif
