// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_OPTIMISEDIRECTIONSET_HEADER
#define RAVL_OPTIMISEDIRECTIONSET_HEADER 1
////////////////////////////////////////////////////////////////////////////
//! author="Robert Crida"
//! lib=Optimisation
//! date="22/4/1998"
//! userlevel=Normal
//! example=testNumOptimise.cc
//! docentry="Ravl.Pattern Recognition.Optimisation.Implementation"
//! rcsid="$Id$"
//! file="Ravl/PatternRec/Optimise/OptimiseSobol.hh"

#include "Ravl/PatternRec/Optimise.hh"
#include "Ravl/SArray1d.hh"

namespace RavlN {
  
  //! userlevel=Develop
  //: DirectionSet .
  
  class OptimiseDirectionSetBodyC: public OptimiseBodyC
  {
  public:
    OptimiseDirectionSetBodyC ();
    //: Default constructor
    //!param: numSamples - number of samples to check
    
    OptimiseDirectionSetBodyC (std::istream &in);
    //: Constructs from stream
    
  protected:
    VectorC MinimalX (const CostC &domain) const;
    //: Determines Xmin=arg min_{X} |f(X)-Yd|
    
    VectorC FindMin(const VectorC &start,const VectorC &dir);
    //: Find minimum along direction 'dir' from start, 
    
    virtual const StringC GetInfo () const;
    //: Prints information about the optimizer
    
    virtual bool Save (std::ostream &out) const;
    //: Writes object to stream, can be loaded using constructor
    
    SArray1dC<VectorC> u;
  };
  
  //! userlevel=Normal
  //: DirectionSet distribution search optimisation.
  //
  // Class for performing a search using a direction set.
  
  class OptimiseDirectionSetC: public OptimiseC
  {
  public:
    OptimiseDirectionSetC ()
      : OptimiseC(*(new OptimiseDirectionSetBodyC (initState)))
    {}
    //: Constructor
    //!param: numSamples - number of samples to check
  };
  
}

#endif

