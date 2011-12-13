// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
////////////////////////////////////////////////////////////////////
//! lib=RavlIO
//! file="Ravl/Core/IO/TypeConverter.cc"

#include "Ravl/DP/TypeConverter.hh"
#include "Ravl/GraphBestRoute.hh"
#include "Ravl/MTLocks.hh"

#define DODEBUG 0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlN {
  
  //: Constructor.
  TypeConverterBodyC::TypeConverterBodyC()
   : m_conversionCache(1000),
     m_version(0)
  {}
  
  //: Get the graph node associated with a named type

  GraphNodeHC<StringC,DPConverterBaseC> TypeConverterBodyC::GetTypeNode(const StringC &name) const {
    const GraphNodeHC<StringC,DPConverterBaseC> *ret = NodeTab().Lookup(name);
    if(ret != 0) return *ret;
    return GraphNodeHC<StringC,DPConverterBaseC>();
  }

  GraphNodeHC<StringC,DPConverterBaseC> TypeConverterBodyC::GetTypeNode(const type_info &inf) const {
    StringC typeName(inf.name());
    return GetTypeNode(typeName);
  }

  //: Get the graph node associated with a type_info, create if needed

  GraphNodeHC<StringC,DPConverterBaseC> TypeConverterBodyC::UseTypeNode(const type_info &inf) {
    StringC typeName(inf.name());
    GraphNodeHC<StringC,DPConverterBaseC> &ret = NodeTab()[typeName];
    if(ret.IsValid())
      return ret;
    ret = ConvGraph().InsNode(typeName);
    return ret;
  }

  //! Find a conversion
  bool TypeConverterBodyC::FindConversion(const type_info &from,
                                          const type_info &to,
                                          RealT &finalCost,
                                          DListC<GraphEdgeIterC<StringC,DPConverterBaseC> > &convResult) const
  {
    if(from == to)
      return true;
#if !(RAVL_COMPILER_VISUALCPP && !RAVL_COMPILER_VISUALCPPNET) || RAVL_COMPILER_VISUALCPPNET_2005
    // Visual C++ can't handle ptr's to functions with reference args.
    // hopefully we'll find a way around this but for now its out.

    DListC<GraphEdgeIterC<StringC,DPConverterBaseC> > conv;

    Tuple2C<StringC,StringC> cacheKey(from.name(),to.name());
    UIntT version;
    {
      MTReadLockC cacheLock(5);
      if(m_conversionCache.Lookup(cacheKey,conv))
        return !conv.IsEmpty();

      GraphNodeHC<StringC,DPConverterBaseC> tFrom = GetTypeNode(cacheKey.Data1());
      GraphNodeHC<StringC,DPConverterBaseC> tTo = GetTypeNode(cacheKey.Data2());
      // Both the source and destination have to be registered for any
      // conversion to exist.
      if(tFrom.IsValid() && tTo.IsValid()) {
        conv = GraphBestRoute(ConvGraph(),
                            tFrom,
                            tTo,
                            finalCost,
                            &TypeConverterBodyC::EdgeEval);
      }

      // Remember the version of the graph we used to generate the conversion.
      version = m_version;
      cacheLock.Unlock();
    }
    {
      MTWriteLockC cacheLock(5);
      if(version == m_version) // Check things haven't changed since we re-locked.
        m_conversionCache.Insert(cacheKey,conv);
      cacheLock.Unlock();
    }
    convResult = conv;
    return !conv.IsEmpty();
#else
    RavlAssert(0);
    return false;
#endif
  }


  //: Test if conversion is possible.
  
  bool TypeConverterBodyC::CanConvert(const type_info &from,const type_info &to) {
    RealT finalCost = 0;
    DListC<GraphEdgeIterC<StringC,DPConverterBaseC> > conv;
    return FindConversion(from,to,finalCost,conv);
  }

  //: Do conversion through abstract handles.
  
  RCAbstractC TypeConverterBodyC::DoConversion(const RCAbstractC &dat,const type_info &from,const type_info &to) {
    if(from == to || !dat.IsValid())
      return dat;
    ONDEBUG(cout << "Asked to convert " << from.name() << " to " << to.name() << endl);
    DListC<GraphEdgeIterC<StringC,DPConverterBaseC> > conv;  
    RealT finalCost = 0;

    if(!FindConversion(from,to,finalCost,conv)) {
      ONDEBUG(cout << "No conversion from " << from.name() << " to " << to.name() << endl);
      return RCAbstractC();
    }
    
    // Do conversion.
    RCAbstractC at = dat;
    for(DLIterC<GraphEdgeIterC<StringC,DPConverterBaseC> > it(conv);it;it++)
      at = it.Data().Data().Apply(at);
    return at;
    
  }
  
  //: Find a conversion.
  // If found the cost of conversion is put into finalCost.
  
  DListC<DPConverterBaseC> TypeConverterBodyC::FindConversion(const type_info &from,const type_info &to,RealT &finalCost)  {
    ONDEBUG(cout << "Asked to convert " << from.name() << " to " << to.name() << endl);
    DListC<GraphEdgeIterC<StringC,DPConverterBaseC> > conv;  
    //  typedef RealT (*AFuncT)(const DPConverterBaseC &);

    if(!FindConversion(from,to,finalCost,conv)) {
      ONDEBUG(cout << "No conversion from " << from.name() << " to " << to.name() << endl);
      return DListC<DPConverterBaseC>(); // Failed to find conversion.
    }
    
    DListC<DPConverterBaseC> ret;
    for(DLIterC<GraphEdgeIterC<StringC,DPConverterBaseC> > it(conv);it.IsElm();it.Next())
      ret.InsLast(it.Data().Data());
    return ret;
  }
  
  
  RealT TypeConverterBodyC::EdgeEval(const DPConverterBaseC &edge)  { 
    ONDEBUG(cout << "Edge cost " << edge.Cost() << " : " << edge.ArgType(0).name() << " " << edge.Output().name() << endl);
    return edge.Cost(); 
  }
  
  
  //: Find a conversion.
  
  DListC<DPConverterBaseC> TypeConverterBodyC::FindConversion(const type_info &from,const type_info &to) { 
    RealT finalCost = -1;
    return FindConversion(from,to,finalCost);
  } 

  bool TypeConverterBodyC::Insert(DPConverterBaseC &tc) {
    MTWriteLockC cacheLock(5);
    m_version++;
    ConvGraph().InsEdge(UseTypeNode(tc.ArgType(0)),UseTypeNode(tc.Output()),tc.Body());
    // Clear the cache, things may have changed.
    m_conversionCache.Empty();
    // FIXME :- Check for duplication ??
    return true;
  }
  
  //: Remove conversion from system.
    
  bool TypeConverterBodyC::Remove(DPConverterBaseC &tc)  {
    // TODO :- This is slow !!
    ONDEBUG(cerr << "Unregistering converter : "<< tc.ArgType(0).name() << " to " << tc.Output().name() << endl);
    MTWriteLockC cacheLock(5);
    m_version++;
    bool ok = false;
    IntT size = 0;
    for(GraphEdgeIterC<StringC,DPConverterBaseC> it(ConvGraph());it.IsElm();it.Next()) {
      if(&it.Data().Body() == &tc.Body()) {
	it.Del();
	ok = true;
	break;
      }
      size++;
    }
    // Clear the cache.
    m_conversionCache.Empty();

    if(!ok)
      cerr << "Failed !!! " << size << "\n";
    return true;
  }
  
  TypeConverterC &SystemTypeConverter() {
    static TypeConverterC tc(true);
    return tc;
  }

}
