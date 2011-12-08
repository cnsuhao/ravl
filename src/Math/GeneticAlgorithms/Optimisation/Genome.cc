// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2011, Charles Galambos
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! lib=RavlGeneticOptimisation
//! author=Charles Galambos
//! docentry=Ravl.API.Math.Genetic.Optimisation

#include "Ravl/Genetic/Genome.hh"
#include "Ravl/Genetic/GenomeConst.hh"
#include "Ravl/Random.hh"
#include "Ravl/OS/SysLog.hh"
#include "Ravl/BListIter.hh"
#include "Ravl/XMLFactoryRegister.hh"
#include "Ravl/TypeName.hh"
#include "Ravl/PointerManager.hh"
#include "Ravl/VirtualConstructor.hh"

#define DODEBUG 0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlN { namespace GeneticN {

  bool GeneVisitorC::Examine(const GeneC &gene)
  {
    return true;
  }

  // ------------------------------------------------

  //! Factory constructor
  GeneTypeC::GeneTypeC(const XMLFactoryContextC &factory)
   : m_name(factory.AttributeString("name",factory.Path().data()).data()),
     m_defaultWeight(static_cast<float>(factory.AttributeReal("weight",1.0)))
  {}

  //! Constructor

  GeneTypeC::GeneTypeC(const std::string &name)
   : m_name(name),
     m_defaultWeight(1.0)
  {}

  //! Load form a binary stream
  GeneTypeC::GeneTypeC(BinIStreamC &strm)
   : RCBodyVC(strm)
  {
    ByteT version = 0;
    strm >> version;
    if(version != 1)
      throw RavlN::ExceptionUnexpectedVersionInStreamC("GeneTypeC");
    strm >> m_name >> m_defaultWeight;
  }

  //! Load form a binary stream
  GeneTypeC::GeneTypeC(std::istream &strm)
  {
    RavlAssertMsg(0,"not implemented");
  }

  //! Save to binary stream
  bool GeneTypeC::Save(BinOStreamC &strm) const
  {
    if(!RavlN::RCBodyVC::Save(strm))
      return false;
    ByteT version = 1;
    strm << version << m_name << m_defaultWeight;
    return true;
  }

  //! Save to binary stream
  bool GeneTypeC::Save(std::ostream &strm) const
  {
    RavlAssertMsg(0,"not implemented");
    return false;
  }

  //! Dump description in human readable form.
  void GeneTypeC::Dump(std::ostream &strm,UIntT indent) const
  {
    strm << Indent(indent) << "Gene:" << Name() << " Weight:" << m_defaultWeight << " Type:" << RavlN::TypeName(typeid(*this));
  }

  //! Add a new entry to the gene
  void GeneTypeC::AddComponent(const std::string &name,const GeneTypeC &geneType)
  {
    RavlAssert(0);
  }

  //! Lookup component
  bool GeneTypeC::LookupComponent(const std::string &name,GeneTypeC::ConstRefT &geneType)
  {
    RavlAssert(0);
    return false;
  }

  //! Create random instance

  void GeneTypeC::Random(SmartPtrC<GeneC> &newValue) const
  {
    RavlAssert(0);
  }

  //! Mutate a gene

  bool GeneTypeC::Mutate(float fraction,const GeneC &original,RavlN::SmartPtrC<GeneC> &newValue) const
  {
    RavlAssert(0);
    return false;
  }

  //! Mutate a gene
  void GeneTypeC::Cross(const GeneC &original1,const GeneC &original2,RavlN::SmartPtrC<GeneC> &newValue) const {
    if(Random1() > 0.5)
      newValue = &original1;
    else
      newValue = &original2;
  }

  static RavlN::TypeNameC g_typePtrGeneType(typeid(GeneTypeC),"RavlN::GeneticN::GeneTypeC");
  static RavlN::TypeNameC g_typePtrGeneTypeRef(typeid(GeneTypeC::RefT),"RavlN::SmartPtrC<RavlN::GeneticN::GeneTypeC>");

  // ------------------------------------------------------------------

  //! Factory constructor
  GeneC::GeneC(const XMLFactoryContextC &factory)
  {
    rThrowBadConfigContextOnFailS(factory,UseChildComponent("Type",m_type),"No type provided");
  }

  //! Constructor
  GeneC::GeneC(const GeneTypeC &theType)
   : m_type(&theType)
  {}

  //! Load form a binary stream
  GeneC::GeneC(BinIStreamC &strm)
   : RCBodyVC(strm)
  {
    ByteT version = 0;
    strm >> version;
    if(version != 1)
      throw RavlN::ExceptionUnexpectedVersionInStreamC("GeneC");
    strm >> ObjIO(m_type);
  }

  //! Load form a binary stream
  GeneC::GeneC(std::istream &strm)
  {
    RavlAssertMsg(0,"not implemented");
  }

  //! Save to binary stream
  bool GeneC::Save(BinOStreamC &strm) const
  {
    if(!RavlN::RCBodyVC::Save(strm))
      return false;
    ByteT version = 1;
    strm << version << ObjIO(m_type);
    return true;
  }


  //! Save to binary stream
  bool GeneC::Save(std::ostream &strm) const
  {
    RavlAssertMsg(0,"not implemented");
    return false;
  }

  //! Save to binary stream
  bool GeneC::Save(RavlN::XMLOStreamC &strm) const
  {
    strm << RavlN::XMLStartTag("Gene") << XMLAttribute("typename",RavlN::TypeName(typeid(*this)))  << XMLAttribute("name",Name().data());
    return true;
  }


  //! Dump description in human readable form.
  void GeneC::Dump(std::ostream &strm,UIntT indent) const {
    strm << Indent(indent) << "Type:" << RavlN::TypeName(typeid(*this)) << " Name:" << Name().data();
  }

  //! Lookup value
  bool GeneC::Lookup(const std::string &name,GeneC::ConstRefT &component) const
  {
    ONDEBUG(RavlSysLogf(SYSLOG_DEBUG,"Base lookup called in %s  ",Name().data()));
    return false;
  }

  //! Add a new entry to the gene
  void GeneC::AddComponent(const std::string &name,const GeneC &newEntry,const GeneTypeC &geneType)
  {
    RavlAssert(0);
  }

  //! Mutate this gene
  bool GeneC::Mutate(float fraction,GeneC::RefT &newOne) const {
    return m_type->Mutate(fraction,*this,newOne);
  }

  //! Cross this gene
  void GeneC::Cross(const GeneC &other,GeneC::RefT &newOne) const {
    m_type->Cross(*this,other,newOne);
  }

  //! Generate an instance of the class.
  void GeneC::Generate(const GeneFactoryC &context,RCWrapAbstractC &handle) const
  {
    RavlSysLogf(SYSLOG_ERR,"Default generate called for %s ",RavlN::TypeName(typeid(*this)));
    RavlAssertMsg(0,"not implemented");
  }

  //! Visit all gene's in tree.
  void GeneC::Visit(GeneVisitorC &visit) const {
    visit.Examine(*this);
  }

  static RavlN::TypeNameC g_typeGene(typeid(GeneC),"RavlN::GeneticN::GeneC");
  static RavlN::TypeNameC g_typePtrGene(typeid(GeneC::RefT),"RavlN::SmartPtrC<RavlN::GeneticN::GeneC>");
  static RavlN::TypeNameC g_typeConstPtrGene(typeid(GeneC::ConstRefT),"RavlN::SmartPtrC<const RavlN::GeneticN::GeneC>");

  static GeneC::ConstRefT ConvertGeneRefToConst(const GeneC::RefT &value)
  { return value.BodyPtr(); }

  DP_REGISTER_CONVERSION(ConvertGeneRefToConst,1.0);


  // ---------------------------------------------------------------------

  GenomeC::GenomeC(const GeneC &rootGene)
   : m_const(false),
     m_genomeRoot(&rootGene),
     m_age(0),
     m_generation(0),
     m_averageCount(0)
  {
  }

  //! Construct a genome from a root gene type.
  GenomeC::GenomeC(const GeneTypeC &rootGeneType)
  : m_const(false),
    m_age(0),
    m_generation(0),
    m_averageCount(0)
  {
    rootGeneType.Random(m_genomeRoot);
  }

  //! Factory constructor
  GenomeC::GenomeC(const XMLFactoryContextC &factory)
   : m_const(factory.AttributeBool("const",false)),
     m_age(0),
     m_generation(0),
     m_averageCount(0)
  {
    factory.UseComponent("Gene",m_genomeRoot);
  }


  //! Load form a binary stream
  GenomeC::GenomeC(BinIStreamC &strm)
   : RCBodyVC(strm),
     m_const(false),
     m_age(0),
     m_averageCount(0)
  {
    ByteT version = 0;
    strm >> version;
    if(version != 1)
      throw RavlN::ExceptionUnexpectedVersionInStreamC("GenomeC");
    strm >> ObjIO(m_genomeRoot) >> m_age >> m_generation;
  }

  //! Load form a binary stream
  GenomeC::GenomeC(std::istream &strm)
   : RCBodyVC(strm),
     m_averageCount(0)
  {
    RavlAssertMsg(0,"not implemented");
  }

  //! Save to binary stream
  bool GenomeC::Save(BinOStreamC &strm) const
  {
    if(!RCBodyVC::Save(strm))
      return false;
    ByteT version = 1;
    strm << version;
    strm << ObjIO(m_genomeRoot) << m_age << m_generation;
    return true;
  }


  //! Save to binary stream
  bool GenomeC::Save(std::ostream &strm) const
  {
    if(m_genomeRoot.IsValid())
      m_genomeRoot->Save(strm);
    return false;
  }

  //! Save to binary stream
  bool GenomeC::Save(RavlN::XMLOStreamC &strm) const {
    strm << RavlN::XMLStartTag("Genome") << XMLAttribute("typename",RavlN::TypeName(typeid(*this))) << XMLContent;
     //      << XMLIndent <<  "Some data... ";
    if(m_genomeRoot.IsValid()) {
      m_genomeRoot->Save(strm);
      strm << XMLIndentDown << XMLEndTag << " ";
    }
    strm << XMLIndentDown << XMLEndTag << " ";
    return true;
  }

  class GeneCounterC
   : public GeneVisitorC
  {
  public:
    GeneCounterC()
     : m_counter(0)
    {}

    // Examine a gene.
    virtual bool Examine(const GeneC &gene) {
      m_counter++;
      return true;
    }

    size_t Count() const
    { return m_counter; }

  protected:
    size_t m_counter;
  };

  //! Access number of gene's in the genome.
  size_t  GenomeC::Size() const {
    GeneCounterC counter;
    m_genomeRoot->Visit(counter);
    return counter.Count();
  }

  //! Mutate this genome.
  bool GenomeC::Mutate(float fraction,GenomeC::RefT &newGenome) const
  {
    GeneC::RefT newRootGene;
    bool ret = false;
    if(fraction <= 1e-6) {
      // We've asked for a near identical copy.
      ret = m_genomeRoot->Mutate(fraction,newRootGene);
    } else {
      int tryNo = 20;
      while(!ret && tryNo-- > 0) {
        ret = m_genomeRoot->Mutate(fraction,newRootGene);
      }
    }
    newGenome = new GenomeC(*newRootGene);
    newGenome->SetAge(m_age+1);
    return ret;
  }

  //! Cross this genome with another
  void GenomeC::Cross(const GenomeC &other,GenomeC::RefT &newGenome) const
  {
    GeneC::RefT newRootGene;
    m_genomeRoot->Cross(other.RootGene(),newRootGene);
    newGenome = new GenomeC(*newRootGene);
    newGenome->SetAge(RavlN::Max(m_age,other.Age()));
  }

  //! Update running average score, return the latest value.
  float GenomeC::UpdateScore(float newScore,UIntT maxAge) {
    if(m_averageCount == 0) {
      m_averageCount = 1;
      m_averageScore = newScore;
      return newScore;
    }
    m_averageScore = (m_averageScore * static_cast<float>(m_averageCount) + newScore) / static_cast<float>(m_averageCount + 1);
    if(m_averageCount < maxAge)
      m_averageCount++;
    return m_averageScore;
  }


  XMLFactoryRegisterC<GenomeC> g_registerGenome("RavlN::GeneticN::GenomeC");
  static RavlN::TypeNameC g_typePtrGenome(typeid(GenomeC::RefT),"RavlN::SmartPtrC<RavlN::GeneticN::GenomeC>");

  // ----------------------------------------------------------------------

  //! Construct from a genome.

  GenomeScaffoldC::GenomeScaffoldC(const GenomeC &genome)
   : m_genome(&genome)
  {}


  // ----------------------------------------------------------------------

  //! Default factory
  GeneFactoryC::GeneFactoryC()
  {}

  //! First level constructor.
  GeneFactoryC::GeneFactoryC(const GenomeC &genome)
   :  m_scaffold(new GenomeScaffoldC(genome))
  {
    genome.UpdateShares(*this);
  }

  //! Push another level on the stack.
  GeneFactoryC::GeneFactoryC(const GeneFactoryC &geneFactory,const GeneC &gene)
   : m_path(geneFactory.m_path),
     m_scaffold(geneFactory.m_scaffold)
  {
    m_path.Push(&gene);
  }

  //! Get the component.
  void GeneFactoryC::GetComponent(const std::string &name,GeneC::ConstRefT &component,const GeneTypeC &geneType) const {
    if(m_path.First()->Lookup(name,component)) {
      ONDEBUG(RavlSysLogf(SYSLOG_DEBUG,"Lookup of '%s' in current context '%s' succeeded .",name.data(),m_path.First()->Name().data()));
      return ;
    }
    ONDEBUG(RavlSysLogf(SYSLOG_DEBUG,"Failed to lookup component '%s' in current context '%s' .",name.data(),m_path.First()->Name().data()));
    ONDEBUG(RavlSysLogf(SYSLOG_DEBUG,"Node: %s ",RavlN::StringOf(m_path.First()).data()));
    {
      if(m_scaffold->IsGenomeConst()) {
        RavlSysLogf(SYSLOG_ERR,"Incomplete genome, can't instantiate.");
        throw RavlN::ExceptionOperationFailedC("Can't instantiate incomplete genome. ");
      }
      GeneC::RefT newComponent;
      geneType.Random(newComponent);
      m_path.First()->AddComponent(name,*newComponent,geneType);
      component = newComponent.BodyPtr();
    }
  }

  void GeneFactoryC::Get(const std::string &name,IntT &value,const GeneTypeC &geneType) const
  {
    GeneC::ConstRefT component;
    GetComponent(name,component,geneType);
    const GeneIntC &theGene = dynamic_cast<const GeneIntC &>(*component);
    value = theGene.Value();
  }

  void GeneFactoryC::Get(const std::string &name,float &value,const GeneTypeC &geneType) const
  {
    GeneC::ConstRefT component;
    GetComponent(name,component,geneType);
    const GeneFloatC &theGene = dynamic_cast<const GeneFloatC &>(*component);
    value = theGene.Value();
  }

  //! Get a real value.
  void GeneFactoryC::Get(const std::string &name,double &value,const GeneTypeC &geneType) const
  {
    GeneC::ConstRefT component;
    GetComponent(name,component,geneType);
    const GeneFloatC &theGene = dynamic_cast<const GeneFloatC &>(*component);
    value = theGene.Value();
  }


  //! Get the root object as an abstract handle
  void GeneFactoryC::Get(RCWrapAbstractC &obj,const type_info &to) const
  {
    RCWrapAbstractC handle;
    m_scaffold->Genome().RootGene().Generate(*this,handle);
    RavlAssert(handle.IsValid());
    obj = SystemTypeConverter().DoConversion(handle.Abstract(),handle.DataType(),to);
    if(!obj.IsValid()) {
      RavlSysLogf(SYSLOG_ERR,"Failed to convert generated type from %s to %s ",RavlN::TypeName(handle.DataType()),RavlN::TypeName(to));
      RavlAssert(0);
      throw RavlN::ExceptionOperationFailedC("Failed to instantiate genome. ");
    }
  }


  //! Check if gene is in stack already.
  bool GeneFactoryC::CheckStackFor(const GeneC &gene) const {
    for(BListIterC<GeneC::RefT> it(m_path);it;it++) {
      if(it.Data().BodyPtr() == &gene)
        return true;
    }
    return false;
  }

  static RavlN::TypeNameC g_typeGeneFactory(typeid(GeneFactoryC),"RavlN::GeneticN::GeneFactoryC");

  std::ostream &operator<<(std::ostream &strm,const GeneFactoryC &factory) {
    strm << "GeneFactoryC.";
    return strm;
  }

  std::istream &operator>>(std::istream &strm,GeneFactoryC &factory)
  {
    RavlAssertMsg(0,"not implemented");
    return strm;
  }

  RavlN::BinOStreamC &operator<<(RavlN::BinOStreamC &strm,const GeneFactoryC &factory) {
    RavlAssertMsg(0,"not implemented");
    return strm;
  }

  RavlN::BinIStreamC &operator>>(RavlN::BinIStreamC &strm,GeneFactoryC &factory){
    RavlAssertMsg(0,"not implemented");
    return strm;
  }

  // Generate a gene factory.

  static GeneFactoryC ConvertGenome2GeneFactory(const GenomeC::RefT &genome)
  { return GeneFactoryC(*genome); }

  //DP_REGISTER_CONVERSION_NAMED(ConvertGenome2GeneFactory,1.0,"ConvertGenome2GeneFactory");
  static RavlN::DPConverterBaseC DPConv_ConvertGenome2GeneFactory(RavlN::RegisterConversion(ConvertGenome2GeneFactory,1.0));


}}

