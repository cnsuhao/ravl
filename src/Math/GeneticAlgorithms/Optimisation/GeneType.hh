// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2011, Charles Galambos
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_GENETIC_GENETYPE_HH
#define RAVL_GENETIC_GENETYPE_HH 1
//! lib=RavlGeneticOptimisation
//! author=Charles Galambos
//! docentry=Ravl.API.Math.Genetic.Optimisation

#include "Ravl/STL.hh"
#include "Ravl/SmartPtr.hh"
#include "Ravl/BStack.hh"
#include "Ravl/Hash.hh"
#include "Ravl/DP/TypeConverter.hh"
#include "Ravl/OS/SysLog.hh"
#include "Ravl/TypeName.hh"

namespace RavlN {
  class XMLFactoryContextC;
}

namespace RavlN { namespace GeneticN {

  class GeneFactoryC;
  class GeneC;
  class GenePaletteC;

  //! Base class for gene visitor

  class GeneVisitorC
  {
  public:
    // Examine a gene.
    virtual bool Examine(const GeneC &gene);
  };

  //! Base class for all gene's

  class GeneTypeC
   : public RavlN::RCBodyVC
  {
  public:
    //! Factory constructor
    GeneTypeC(const XMLFactoryContextC &factory);

    //! Constructor
    GeneTypeC(const std::string &name);

    //! Load form a binary stream
    GeneTypeC(BinIStreamC &strm);

    //! Load form a binary stream
    GeneTypeC(std::istream &strm);

    //! Save to binary stream
    virtual bool Save(BinOStreamC &strm) const;

    //! Save to binary stream
    virtual bool Save(std::ostream &strm) const;

    //! Dump description in human readable form.
    virtual void Dump(std::ostream &strm,UIntT indent = 0) const;

    //! Access name of gene
    const std::string &Name() const
    { return m_name; }

    //! Add a new entry to the gene
    virtual void AddComponent(const std::string &name,const GeneTypeC &geneType);

    //! Lookup component
    virtual bool LookupComponent(const std::string &name,RavlN::SmartPtrC<const GeneTypeC> &geneType);

    //! Create random instance
    virtual void Random(RavlN::SmartPtrC<GeneC> &newValue) const = 0;

    //! Mutate a gene
    virtual bool Mutate(float fraction,const GeneC &original,RavlN::SmartPtrC<GeneC> &newValue) const = 0;

    //! Mutate a gene
    virtual void Cross(const GeneC &original1,const GeneC &original2,RavlN::SmartPtrC<GeneC> &newValue) const;

    //! Access default weight for gene
    //! Used to effect relative frequencies in meta types.
    //! If unmodified, will be 1.
    float DefaultWeight() const
    { return m_defaultWeight; }

    //! Set default weight for gene
    //! Used to effect relative frequencies in meta types.
    //! If unmodified, will be 1.
    void SetDefaultWeight(float value)
    { m_defaultWeight = value; }

    // Reference to this gene.
    typedef RavlN::SmartPtrC<GeneTypeC> RefT;

    // Const reference to this gene.
    typedef RavlN::SmartPtrC<const GeneTypeC> ConstRefT;

  protected:
    std::string m_name;
    float m_defaultWeight;
  };

  //! Gene for a single block.

  class GeneC
   : public RavlN::RCBodyVC
  {
  public:
    //! Factory constructor
    GeneC(const XMLFactoryContextC &factory);

    //! Constructor
    GeneC(const GeneTypeC &theType);

    //! Load form a binary stream
    GeneC(BinIStreamC &strm);

    //! Load form a binary stream
    GeneC(std::istream &strm);

    //! Save to binary stream
    virtual bool Save(BinOStreamC &strm) const;

    //! Save to binary stream
    virtual bool Save(std::ostream &strm) const;

    //! Save to binary stream
    virtual bool Save(RavlN::XMLOStreamC &strm) const;

    //! Dump description in human readable form.
    virtual void Dump(std::ostream &strm,UIntT indent = 0) const;

    //! Access name of gene.
    const std::string &Name() const
    { return m_type->Name(); }

    //! Access the gene type.
    const GeneTypeC &Type() const
    { return *m_type; }

    // Reference to this gene.
    typedef RavlN::SmartPtrC<GeneC> RefT;

    // Const reference to this gene.
    typedef RavlN::SmartPtrC<const GeneC> ConstRefT;

    //! Mutate this gene
    bool Mutate(float faction,GeneC::RefT &newOne) const;

    //! Cross this gene
    void Cross(const GeneC &other,GeneC::RefT &newOne) const;

    //! Generate an instance of the class.
    virtual void Generate(const GeneFactoryC &context,RCWrapAbstractC &handle) const;

    //! Lookup value
    virtual bool Lookup(const std::string &name,GeneC::ConstRefT &component) const;

    //! Add a new entry to the gene
    virtual void AddComponent(const std::string &name,const GeneC &newEntry,const GeneTypeC &geneType);

    //! Visit all gene's in tree.
    virtual void Visit(GeneVisitorC &visit) const;

  protected:
    GeneTypeC::RefT m_type;
  };


}}

#endif
