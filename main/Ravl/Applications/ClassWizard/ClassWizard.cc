// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2004, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlClassWizard

#include "Ravl/SourceTools/ClassWizard.hh"
#include "Ravl/SourceTools/SourceCodeManager.hh"
#include "Ravl/CxxDoc/CxxElements.hh"
#include "Ravl/CallMethods.hh"

#define DODEBUG 0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlN {

  //: Constructor.
  
  ClassWizardBodyC::ClassWizardBodyC(const StringC &nrootDir,const StringC &nlocalDir,bool nverbose) 
    : rootDir(nrootDir),
      localDir(nlocalDir),
      parseTree(true),
      verbose(nverbose)
  { parseTree.SetRootFilename(rootDir); }
  
  //: Gather information about the source code.
  
  bool ClassWizardBodyC::GatherInfo() {
    SourceCodeManagerC scm(rootDir);
    ClassWizardC me(*this);
    scm.ForAllDirs(Trigger(me,&ClassWizardC::GatherDir,StringC(),DefsMkFileC()));
    return true;
  }
  
  //: Gather info from file.
  
  bool ClassWizardBodyC::GatherFile(StringC &filename) {
    //cerr << "ClassWizardBodyC::GatherFile(), " << filename << "\n";
    parseTree.Parse(filename);
    return true;
  }
  
  //: Gather info from a directory.
  
  bool ClassWizardBodyC::GatherDir(StringC &dirname,DefsMkFileC &defsmk) {
    ONDEBUG(cerr << "ClassWizardBodyC::GatherDir(), " << dirname << "\n");
    StringListC hdrfiles = defsmk.Value("HEADERS");
    for(DLIterC<StringC> it(hdrfiles);it;it++) {
      FilenameC sfile = dirname + '/' + *it;
      if(!sfile.Exists())
	cerr << "WARNING: Can't find file '" << sfile << "'\n";
      GatherFile(sfile);
    }
    return true;
  }
  
  //: Go through and update local source files.
  
  bool ClassWizardBodyC::ApplyWizard() {
    if(verbose)
      cerr << "Gathering class information. \n";
    
    if(!GatherInfo())
      return false;
    
    if(verbose)
      cerr << "Resolving refrences. \n";
    
    parseTree.Resolve();

    if(verbose)
      cerr << "Scanning for big objects. \n";
    
    ScopeC scope(parseTree.Data());
    //scope.Dump(cout);
    
    ScanScope(scope);
    
    if(verbose)
      cerr << "Saving edits. \n";
    for(HashIterC<StringC,TextFileC> it(textFiles);it;it++) {
      if(!modifiedPrefix.IsEmpty())
	it.Data().SetFilename(modifiedPrefix + it.Key());
      if(!it.Data().Save()) {
	cerr << "ERROR: Failed to save '" << it.Data().Filename() << "'\n";
      }
    }
    return true;
  }

  //: Locate insert point after object.
  
  SourceCursorC ClassWizardBodyC::InsertPoint(TextBufferC &buff,ObjectC &object) {
    SourceCursorC sc(buff);
    
    return sc;
  }
  
  //: Append comment to text buffer.
  
  bool ClassWizardBodyC::WriteComment(SourceCursorC &sc,ObjectC &obj,bool markAsAuto) {
    if(!obj.Comment().Header().IsEmpty())
      sc += StringC("//:") + obj.Comment().Header();
    if(!obj.Comment().Text().IsEmpty())
      sc += StringC("//") + obj.Comment().Text();
    if(markAsAuto)
      sc += "//!cwiz:author";
    return true;
  }
  
  //: Write the handle method for the given method.
  
  bool ClassWizardBodyC::WriteHandleMethod(SourceCursorC &sc,ObjectC &obj) {
    RCHashC<StringC,ObjectC> templSub;
    
    sc += obj.FullName(templSub,descGen);
    MethodC method(obj);
    StringC funcArgs;
    DLIterC<DataTypeC> it(method.Args());
    if(it) {
      funcArgs += it->Alias();
      for(it++;it;it++)
	funcArgs += StringC(",") + it->Alias();
    }
    StringC retType = method.ReturnType().FullName(templSub,descGen);
    if(retType != "void")
      sc += StringC("{ return Body().") + obj.Var("BaseName") +"(" + funcArgs + "); }";
    else
      sc += StringC("{ Body().") + obj.Var("BaseName") +"(" + funcArgs + "); }";
    WriteComment(sc,obj);
    return true;
  }
  
  //: Write the handle constructor.
  
  bool ClassWizardBodyC::WriteHandleConstructor(SourceCursorC &sc,ObjectC &obj,const StringC &handleBaseClass) {
    ONDEBUG(cerr << "ClassWizardBodyC::WriteHandleConstructor(). \n");
    RCHashC<StringC,ObjectC> templSub;
    MethodC method(obj);
    StringC funcArgs;
    DLIterC<DataTypeC> it(method.Args());
    if(it) {
      funcArgs += it->Alias();
      for(it++;it;it++)
	funcArgs += StringC(",") + it->Alias();
    }
    StringC fullName= obj.FullName(templSub,descGen);
    StringC baseName=obj.Var("BaseName");
    StringC className = baseName.Copy();
    className.gsub("BodyC","C");
    fullName.gsub(baseName,className);
    
    ONDEBUG(cerr << "baseName=" << baseName << " className=" << className << "\n");
    
    sc += fullName;
    sc.AddIndent(1);
    sc += StringC(": ") + handleBaseClass + "(*new " + baseName + "(" + funcArgs + "))";
    sc.AddIndent(-1);
    sc += "{}";
    WriteComment(sc,obj);
    return true;
  }
  
  //: Generate text for handle class.
  
  bool ClassWizardBodyC::WriteHandleClass(SourceCursorC &sc,ObjectC &bodyObj,const StringC &className) {
    TextBufferC textBuff(true);
    
    ClassC bodyClass(bodyObj);
    // Assume we're in the right namespace.
    
    sc += "//! userlevel=normal";
    if(bodyObj.Comment().Header().IsEmpty())
      sc += StringC("//: Handle for ") + bodyClass.Name();
    WriteComment(sc,bodyObj);
    sc += "";
    sc += StringC("class ") + className;
    
    // Work out class inheritance.
    // What does body class inherit from ?
    DListC<StringC> baseClasses;
    for(DLIterC<ObjectC> it(bodyClass.Uses());it;it++) {
      StringC iname = it->Var("classname");
      //cerr << "Inherit=" << iname << "\n";
      if(iname == "RCBodyC" || iname == "RCBodyVC") {
	baseClasses.InsLast(StringC("RCHandleC<") + bodyObj.Name() + StringC(">"));
	continue;
      }
      if(!iname.contains("BodyC",-1))
	continue; // Ingore inheritance from non-big objects.
      StringC handleName = iname.before("BodyC");
      baseClasses.InsLast(handleName);
    }
    
    //: Write inheritance info into class.
    sc.AddIndent(1);
    bool gotOne = false;
    for(DLIterC<StringC> it(baseClasses);it;it++) {
      if(!gotOne) {
	sc += StringC(": public ") + *it;
	gotOne = true;
      } else
	sc += StringC(", public ") + *it;
    }
    sc.AddIndent(-1);
    sc += "{";
    sc += "public:";
    sc.AddIndent(1);
    
    StringC mainBaseClass;
    if(baseClasses.IsEmpty())
      mainBaseClass = StringC("(Unknown)");
    else
      mainBaseClass = baseClasses.First();
    
    StringC rootBaseClass = mainBaseClass;
    
    for(InheritIterC it(bodyClass,SAPublic,true);it;it++) {
      if(!MethodC::IsA(*it)) // Only interested in methods here
	continue;
      if(it->Var("constructor") == "true")
	WriteHandleConstructor(sc,*it,mainBaseClass);
      else
	WriteHandleMethod(sc,*it);
      sc += "";
    }
    sc.AddIndent(-1);
    sc += "protected:\n";
    sc.AddIndent(1);
    
    //: Write body constructor.
    sc += className + "(" + mainBaseClass + " &bod)";
    sc += StringC(" : ") + mainBaseClass + "(bod)";
    sc += "{}";
    sc += "";
    
    //: Write body access.
    sc += bodyObj.Name() + "& Body()";
    sc += StringC("{ return static_cast<") + bodyObj.Name() + " &>(" + rootBaseClass + "::Body()) }" ;
    sc += "//: Body Access. ";
    sc += "";
    
    //: Write const body access.
    sc += StringC("const ") + bodyObj.Name() + "& Body() const" ;
    sc += StringC("{ return static_cast<const ") + bodyObj.Name() + " &>(" + rootBaseClass + "::Body()) }" ;
    sc += "//: Body Access. ";
    sc += "";

    //: Close class.
    
    sc.AddIndent(-1);
    sc += "}";
    
    return true;
  }
  
  //: Get a text file for editing.
  
  TextFileC ClassWizardBodyC::TextFile(const StringC &filename) {
    TextFileC ret;
    if(textFiles.Lookup(filename,ret))
      return ret;
    FilenameC fn(filename);
    if(!fn.Exists()) {
      cerr << "ERROR: File '" << filename << "' not found. \n";
      return ret;
    }
    ret = TextFileC(filename);
    textFiles[filename] = ret;
    return ret;
  }
  
  //: Apply to a directory.
  
  bool ClassWizardBodyC::ApplyClass(ScopeC &scope,ObjectC &bodyObj) {
    StringC handleClassname = bodyObj.Name().before("BodyC") + 'C';
    ONDEBUG(cerr << "ClassWizardBodyC::ApplyClass(), Scope=" << scope.Name() << " Handle=" << handleClassname << " Body=" << bodyObj.Name() << "\n");
    ONDEBUG(cerr << "ClassWizardBodyC::ApplyClass(), File='" << scope.Var("filename") << "'\n");
    ObjectC rawHandleObj = scope.Lookup(handleClassname);
    ClassC bodyClass(bodyObj);
    if(!rawHandleObj.IsValid()) {
      ONDEBUG(cerr << "Failed to find handle class '" << handleClassname << "'\n");
      StringC localFile = bodyObj.Var("filename");
      IntT insertPoint = bodyObj.EndLineno() + 1;
      
      // Write one.
      cerr << "Adding class " << handleClassname << " to " << localFile << " at " << insertPoint << "\n";
      TextFileC txt = TextFile(localFile);
      SourceCursorC sc(txt,1);
      if(!sc.FindLine(insertPoint)) {
	cerr << "ERROR: Failed to find insert point " << insertPoint  << ", skipping edit. \n";
	return false;
      }
      sc += "";
      WriteHandleClass(sc,bodyObj,handleClassname);
      return true;
    }
    if(!ClassC::IsA(rawHandleObj)) {
      cerr <<"WARNING: Unexpected handle type '" << rawHandleObj.TypeName() << "'. \n";
      return false;
    } 
    ClassC handleObj = rawHandleObj;
    ONDEBUG(cerr << "EndLine = " << handleObj.EndLineno() << " " << handleObj.StartLineno()<< "\n");
    
    // Look through handle for cruft that can be removed.
    
    for(InheritIterC it(handleObj,SAPublic,true);it;it++) {
      if(!MethodC::IsA(*it)) // Only interested in methods here
	continue;
      if(verbose)
	cerr << "Checking method " << it->Name() << "\n";
      //it->Dump(cerr);
      if(it->Comment().Locals()["author"].TopAndTail() != "cwiz")
	continue; // Only modified methods we wrote
      bool isConstructor = false;
      if(it->Var("constructor") == "true")
	isConstructor = true;
      StringC bodyName = it->Name();
      if(isConstructor)
	continue; // Ignore for now.
      ObjectC bodyMethod = bodyClass.Lookup(bodyName);
      if(bodyMethod.IsValid()) {
	//cerr << " found body " << bodyMethod.Name() << "\n";
	continue; // Its fine.
      }
      StringC localFile = bodyObj.Var("filename");
      // Can't find corresponding body method, must have been deleted.
      TextFileC txt = TextFile(localFile);
      if(!txt.IsValid())
	continue; // Couldn't find file.
      // Work out what exactly to delete.
      IntT firstLine = it->StartLineno();
      IntT lastLine = it->EndLineno();
      if(it->Comment().EndLine() > lastLine)
	lastLine = it->Comment().EndLine();
      SourceCursorC sc(txt,2);
      // Check for blank line after method to remove.
      if(sc.FindLine(lastLine+1) && sc.IsBlank()) 
	lastLine++;
      cerr << "Deleteing method " << it->Name() << " from " << localFile << " at lines " << firstLine << " to " << lastLine << "\n";
      // Do the dirty deed.
      txt.Delete(firstLine,lastLine);
    }
    
    
    
    // Now we've got both the handle and the body classes, lets look for public methods that aren't in the handle.
    // Note we're only iterating through the public part of the local scope, .
    
    StringC mainBaseClass = "Unknown";
    
    int endOfLastHandle=-1;
    for(InheritIterC it(bodyObj,SAPublic,true);it;it++) {
      if(!MethodC::IsA(*it)) // Only interested in methods here
	continue;
      ONDEBUG(cerr << "Found method '" << it->Name() << "' \n");
      bool isConstructor = false;
      if(it->Var("constructor") == "true")
	isConstructor = true;
      StringC handleName = it->Name();
      if(isConstructor)
	continue; // Ignore for now.
      ObjectC handleMethod = handleObj.Lookup(handleName);

      // Does method exist ?
      
      if(!handleMethod.IsValid()) {
	//cerr << "Failed to find handle method. \n";
	if(endOfLastHandle < 0) {
	  // Need to find were to write it....
	  cerr << "WARNING: Don't know where to write method. skipping. \n";
	  continue;
	}
	StringC localFile = bodyObj.Var("filename");
	cerr << "Adding method " << it->Name() << " to " << localFile << " at " << (endOfLastHandle+1) << "\n";
	TextFileC txt = TextFile(localFile);
	SourceCursorC sc(txt,2);
	if(!sc.FindLine(endOfLastHandle+1)) {
	  cerr << "ERROR: Failed to find insert point " << endOfLastHandle << ", skipping edit. \n";
	  continue;
	}
	sc += "";
	if(isConstructor)
	  WriteHandleConstructor(sc,*it,mainBaseClass);
	else
	  WriteHandleMethod(sc,*it);
	continue;
      }
      
      // FIXME: Are the comments right ?
      
      // Make a note of where we are.
      
      endOfLastHandle = handleMethod.Comment().EndLine();
      if(endOfLastHandle < 0)
	endOfLastHandle = handleMethod.EndLineno();
      
      //cerr << "EndLine = " << handleMethod.EndLineno() << " " << handleMethod.StartLineno()<< " Comment=" << handleMethod.Comment().StartLine() << " "<<handleMethod.Comment().EndLine() << " \n";
    }
    return true;
  }
  
  //: Scan scope
  
  bool ClassWizardBodyC::ScanScope(ScopeC &scope) {
    ONDEBUG(cerr << "ClassWizardBodyC::ScanScope(), Scan scope " << scope.Name() << "\n");
    for(DLIterC<ObjectC> it(scope.List());it;it++) {
      if(ClassC::IsA(*it)) { // Got a class ?
	if(it->Name().contains("BodyC",-1)) {
	  ONDEBUG(cerr << "Found body class '" << it->Name() << "\n");
	  ApplyClass(scope,*it);
	}
	continue; // Don't look inside class scopes.
      }
      if(ScopeC::IsA(*it)) {
	ScopeC subScope(*it);
	if(!ScanScope(subScope))
	  return false;
      }
    }
    
    return true;
  }
  
}
  
