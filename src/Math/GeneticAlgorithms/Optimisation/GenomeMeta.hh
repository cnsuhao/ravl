// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2011, Charles Galambos
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_GENETIC_GENOMEMETA_HH
#define RAVL_GENETIC_GENOMEMETA_HH 1
//! lib=RavlGeneticOptimisation
//! author=Charles Galambos
//! docentry=Ravl.API.Math.Genetic.Optimisation

#include "Ravl/Genetic/Genome.hh"

namespace RavlN { namespace GeneticN {

  //! Gene type which is an enumeration of its possible values.

  class GeneTypeEnumC
   : public GeneTypeC
  {
  public:
    //! Factory constructor
    GeneTypeEnumC(const XMLFactoryContextC &factory);

    //! Constructor
    GeneTypeEnumC(const std::string &name,const std::vector<GeneC::ConstRefT> &values);

    //! Load form a binary stream
    GeneTypeEnumC(BinIStreamC &strm);

    //! Load form a binary stream
    GeneTypeEnumC(std::istream &strm);

    //! Save to binary stream
    virtual bool Save(BinOStreamC &strm) const;

    //! Save to binary stream
    virtual bool Save(std::ostream &strm) const;

    //! Create randomise value
    virtual void Random(GeneC::RefT &newValue) const;

    //! Mutate a gene
    virtual bool Mutate(float fraction,const GeneC &original,RavlN::SmartPtrC<GeneC> &newValue) const;

    // Reference to this gene.
    typedef RavlN::SmartPtrC<GeneTypeEnumC> RefT;

    // Const reference to this gene.
    typedef RavlN::SmartPtrC<const GeneTypeEnumC> ConstRefT;

  protected:
    std::vector<GeneC::ConstRefT> m_values;
  };

   //! Gene type which is an enumeration of its possible types.
   //! Where each is given an equal chance of occurring.

   class GeneTypeMetaC
    : public GeneTypeC
   {
   public:
     //! Factory constructor
     GeneTypeMetaC(const XMLFactoryContextC &factory);

     //! Constructor
     GeneTypeMetaC(const std::string &name,const std::vector<GeneTypeC::ConstRefT> &types);

     //! Load form a binary stream
     GeneTypeMetaC(BinIStreamC &strm);

     //! Load form a binary stream
     GeneTypeMetaC(std::istream &strm);

     //! Save to binary stream
     virtual bool Save(BinOStreamC &strm) const;

     //! Save to binary stream
     virtual bool Save(std::ostream &strm) const;

     //! Create randomise value
     virtual void Random(GeneC::RefT &newValue) const;

     //! Mutate a gene
     virtual bool Mutate(float fraction,const GeneC &original,RavlN::SmartPtrC<GeneC> &newValue) const;

     //! Access list of types
     const std::vector<GeneTypeC::ConstRefT> &Types() const
     { return m_types; }

     //! Add type to list
     //! Setting the weight to a negative number causes the default to be used.
     virtual void AddType(const GeneTypeC &geneType,float weight = -1.0);

     // Reference to this gene.
     typedef RavlN::SmartPtrC<GeneTypeMetaC> RefT;

     // Const reference to this gene.
     typedef RavlN::SmartPtrC<const GeneTypeMetaC> ConstRefT;

   protected:
     std::vector<GeneTypeC::ConstRefT> m_types;
   };


}}

#endif