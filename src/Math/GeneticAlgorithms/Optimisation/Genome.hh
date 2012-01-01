// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2011, Charles Galambos
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_GENETIC_GENOME_HH
#define RAVL_GENETIC_GENOME_HH 1
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

  //! Static Genome for an agent.

  class GenomeC
  : public RavlN::RCBodyVC
  {
  public:
    //! Default constructor
    GenomeC();

    //! Factory constructor
    GenomeC(const XMLFactoryContextC &factory);

    //! Construct a genome from a root gene.
    GenomeC(const GeneC &rootGene);

    //! Construct a genome from a root gene type.
    GenomeC(const GeneTypeC &rootGeneType);

    //! Load form a binary stream
    GenomeC(BinIStreamC &strm);

    //! Load form a binary stream
    GenomeC(std::istream &strm);

    //! Save to binary stream
    virtual bool Save(BinOStreamC &strm) const;

    //! Save to binary stream
    virtual bool Save(std::ostream &strm) const;

    //! Save to binary stream
    virtual bool Save(RavlN::XMLOStreamC &strm) const;

    //! Set const.
    void SetConst(bool asConst)
    { m_const = asConst; }

    //! Test if genome is const.
    bool IsConst() const
    { return m_const; }

    //! Access the root gene.
    const GeneC &RootGene() const
    { return *m_genomeRoot; }

    //! Access number of gene's in the genome.
    size_t Size() const;

    //! Handle to genome
    typedef RavlN::SmartPtrC<GenomeC> RefT;

    //! Mutate this genome.
    bool Mutate(float faction,GenomeC::RefT &newGenome) const;

    //! Cross this genome with another
    void Cross(const GenomeC &other,GenomeC::RefT &newGenome) const;

    //! Update shared information.
    void UpdateShares(GeneFactoryC &factory) const;

    //! Age.
    UIntT Age() const
    { return m_age; }

    UIntT Generation() const
    { return m_generation; }

    //! Set age.
    void SetAge(UIntT value)
    { m_age = value; }

    //! Set the generation
    void SetGeneration(UIntT value)
    { m_generation = value; }

    //! Update running average score, return the latest value.
    float UpdateScore(float newScore,UIntT maxAge);
  protected:

    bool m_const;
    GeneC::RefT m_genomeRoot;
    UIntT m_age;
    UIntT m_generation;

    float m_averageScore;
    UIntT m_averageCount;
  };

  //! Information used in instantiating an agent.

  class GenomeScaffoldC
   : public RavlN::RCBodyC
  {
  public:
    //! Construct from a genome.
    GenomeScaffoldC(const GenomeC &genome);

    //! Access genome associated with scaffold.
    const GenomeC &Genome() const
    { return *m_genome; }

    //! Test if genome is const.
    bool IsGenomeConst() const
    { return m_genome->IsConst(); }

    //! Are we allowed to update the gene ?
    bool AllowUpdate() const
    { return m_allowUpdate; }

    //! Lookup instance of class data.
    bool Lookup(const void *gene,RCAbstractC &data) const
    { return m_parts.Lookup(gene,data); }

    //! Insert into table
    bool Insert(const void *gene,const RCAbstractC &data)
    { return m_parts.Insert(gene,data); }

    //! Lookup instance of class data.
    bool LookupOverride(const GeneC &gene,GeneC::RefT &theGene) const
    { return m_overrides.Lookup(&gene,theGene); }

    //! Insert into table
    bool InsertOverride(const GeneC &gene,const GeneC &data)
    { return m_overrides.Insert(&gene,&data); }

    //! Handle to scaffold
    typedef RavlN::SmartPtrC<GenomeScaffoldC> RefT;
  protected:
    GenomeC::RefT m_genome;
    bool m_allowUpdate;
    RavlN::HashC<const void *,RCAbstractC> m_parts;
    RavlN::HashC<GeneC::RefT,GeneC::RefT> m_overrides;
  };

  //!  Default values for basic types
  const GeneTypeC &GeneType(IntT value);

  //!  Default values for basic types
  const GeneTypeC &GeneType(float value);

  //! Generate a gene type for
  const GeneTypeC &CreateGeneType(const std::type_info &ti);

  //!  Default values for basic types
  template<typename DataT>
  const GeneTypeC &GeneType(DataT &value)
  {
    static GeneTypeC::RefT gt = &CreateGeneType(typeid(DataT));
    return *gt;
  }


  //! Factory class.
  class GeneFactoryC
  {
  public:
    //! Default factory
    GeneFactoryC();

    //! First level constructor.
    GeneFactoryC(const GenomeC &genome);

    //! Push another level on the stack.
    GeneFactoryC(const GeneFactoryC &geneFactory,const GeneC &gene);

  protected:
    //! Get the component.
    void GetComponent(const std::string &name,GeneC::ConstRefT &component,const GeneTypeC &geneType) const;

  public:
    //! Get an integer.
    void Get(const std::string &name,IntT &value,const GeneTypeC &geneType) const;

    //! Get a real value.
    void Get(const std::string &name,float &value,const GeneTypeC &geneType) const;

    //! Get a real value.
    void Get(const std::string &name,double &value,const GeneTypeC &geneType) const;

    //! Get a sub class
    template<typename ValueT>
    void Get(const std::string &name,ValueT &value,const GeneTypeC &geneType) const {
      GeneC::ConstRefT theGene;
      GetComponent(name,theGene,geneType);
      RCWrapAbstractC handle;
      theGene->Generate(*this,handle);
      RavlAssert(handle.IsValid());
      if(!SystemTypeConverter().TypeConvert(handle,value)) {
        RavlSysLogf(SYSLOG_ERR,"Failed to convert generated type from %s to %s ",RavlN::TypeName(handle.DataType()),RavlN::TypeName(typeid(ValueT)));
        RavlAssert(0);
        throw RavlN::ExceptionOperationFailedC("Failed to instantiate genome. ");
      }
    }

    //! Get a sub class
    template<typename ValueT>
    void Get(const std::string &name,ValueT &value) const
    { Get(name,value,GeneType(value)); }

    //! Get the root object as an abstract handle
    void Get(RCWrapAbstractC &obj,const type_info &to) const;

    //! Get the root object
    template<typename ValueT>
    void Get(ValueT &value) const {
      RCWrapC<ValueT> handle(handle);
      Get(handle,typeid(ValueT));
      value = handle.Data();
    }

    //! Lookup instance of class data.
    bool Lookup(const void *gene,RCAbstractC &data) const
    { return m_scaffold->Lookup(gene,data); }

    //! Insert into table
    bool Insert(const void *gene,const RCAbstractC &data) const
    { return m_scaffold->Insert(gene,data); }

    //! Lookup instance of class data.
    bool LookupOverride(const GeneC &gene,GeneC::RefT &theGene) const
    { return m_scaffold->LookupOverride(gene,theGene); }

    //! Insert into table
    bool InsertOverride(const GeneC &gene,const GeneC &data)
    { return m_scaffold->InsertOverride(gene,data); }

    //! Check if gene is in stack already.
    bool CheckStackFor(const GeneC &gene) const;
  protected:
    mutable RavlN::BStackC<GeneC::RefT> m_path;
    mutable GenomeScaffoldC::RefT m_scaffold;
  };

  std::ostream &operator<<(std::ostream &strm,const GeneFactoryC &factory);
  std::istream &operator>>(std::istream &strm,GeneFactoryC &factory);
  RavlN::BinOStreamC &operator<<(RavlN::BinOStreamC &strm,const GeneFactoryC &factory);
  RavlN::BinIStreamC &operator>>(RavlN::BinIStreamC &strm,GeneFactoryC &factory);
}}

#endif
