// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2011, Charles Galambos
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#include "Ravl/Genetic/GenomeConst.hh"
#include "Ravl/Random.hh"
#include "Ravl/OS/SysLog.hh"
#include "Ravl/XMLFactoryRegister.hh"
#include "Ravl/TypeName.hh"
#include "Ravl/PointerManager.hh"
#include "Ravl/VirtualConstructor.hh"
//! lib=RavlGeneticOptimisation
//! author=Charles Galambos
//! docentry=Ravl.API.Math.Genetic.Optimisation

namespace RavlN { namespace GeneticN {

  // ------------------------------------------------------------------

  //! Factory constructor
  GeneTypeIntC::GeneTypeIntC(const XMLFactoryContextC &factory)
   : GeneTypeC(factory),
     m_min(factory.AttributeInt("min",0)),
     m_max(factory.AttributeInt("max",100))
  {}

  //! Constructor
  GeneTypeIntC::GeneTypeIntC(const std::string &name,IntT min,IntT max)
  : GeneTypeC(name),
    m_min(min),
    m_max(max)
  {}

  //! Load form a binary stream
  GeneTypeIntC::GeneTypeIntC(BinIStreamC &strm)
   : GeneTypeC(strm)
  {
    ByteT version = 0;
    strm >> version;
    if(version != 1)
      throw RavlN::ExceptionUnexpectedVersionInStreamC("GeneTypeIntC");
    strm >> m_min >> m_max;
  }

  //! Load form a binary stream
  GeneTypeIntC::GeneTypeIntC(std::istream &strm)
   : GeneTypeC(strm)
  {
    RavlAssertMsg(0,"not implemented");
  }

  //! Save to binary stream
  bool GeneTypeIntC::Save(BinOStreamC &strm) const
  {
    if(!GeneTypeC::Save(strm))
      return false;
    ByteT version = 1;
    strm << version << m_min << m_max;
    return true;
  }


  //! Save to binary stream
  bool GeneTypeIntC::Save(std::ostream &strm) const
  {
    strm << " GeneTypeInt " << m_min << " - " << m_max;
    return false;
  }


  //! Create randomise value
  void GeneTypeIntC::Random(GeneC::RefT &newGene) const
  {
    float value = RavlN::Random1() * static_cast<float>(m_max - m_min) + static_cast<float>(m_min);
    IntT newValue = Floor(value);
    if(newValue < m_min)
       newValue = m_min;
     if(newValue > m_max)
       newValue = m_max;
     newGene = new GeneIntC(*this,newValue);
  }

  //! Mutate a gene
  bool GeneTypeIntC::Mutate(float fraction,const GeneC &original,RavlN::SmartPtrC<GeneC> &newGene) const
  {
    if(fraction <= 0) {
      newGene = &original;
      return false;
    }
    const GeneIntC &originalInt = dynamic_cast<const GeneIntC &>(original);

    float value = static_cast<float>(RavlN::Random1() * (m_max - m_min)) + static_cast<float>(m_min);
    IntT newValue = Floor(fraction * static_cast<float>(originalInt.Value()) + (1.0 - fraction) * value);
    // Clip to valid range just in case of rounding problems
    if(newValue < m_min)
      newValue = m_min;
    if(newValue > m_max)
      newValue = m_max;
    newGene = new GeneIntC(*this,newValue);
    return true;
  }

  //!  Default values for basic types
  const GeneTypeC &GeneType(IntT value) {
    static GeneTypeC::RefT defaultIntType = new GeneTypeIntC("DefaultInt",0,100);
    return *defaultIntType;
  }

  XMLFactoryRegisterConvertC<GeneTypeIntC,GeneTypeC> g_registerGeneTypeInt("RavlN::GeneticN::GeneTypeIntC");
  RAVL_INITVIRTUALCONSTRUCTOR_NAMED(GeneTypeIntC,"RavlN::GeneticN::GeneTypeIntC");
  static RavlN::TypeNameC g_typePtrGeneTypeInt(typeid(GeneTypeIntC::RefT),"RavlN::SmartPtrC<RavlN::GeneticN::GeneTypeIntC>");

  // ------------------------------------------------------------------

  //! Factory constructor
  GeneIntC::GeneIntC(const XMLFactoryContextC &factory)
   : GeneC(factory),
     m_value(factory.AttributeInt("value",0))
  {
    RavlAssertMsg(dynamic_cast<GeneTypeIntC *>(m_type.BodyPtr()) != 0,"Wrong type");
  }

  GeneIntC::GeneIntC(const GeneTypeC &geneType,IntT value)
  : GeneC(geneType),
    m_value(value)
  {}

  //! Load form a binary stream
  GeneIntC::GeneIntC(BinIStreamC &strm)
   : GeneC(strm)
  {
    ByteT version = 0;
    strm >> version;
    if(version != 1)
      throw RavlN::ExceptionUnexpectedVersionInStreamC("GeneC");
    strm >> m_value;
  }

  //! Load form a binary stream
  GeneIntC::GeneIntC(std::istream &strm)
   : GeneC(strm)
  {
    RavlAssertMsg(0,"not implemented");
  }

  //! Save to binary stream
  bool GeneIntC::Save(BinOStreamC &strm) const
  {
    if(!GeneC::Save(strm))
      return false;
    ByteT version = 1;
    strm << version << m_value;
    return true;
  }


  //! Save to binary stream
  bool GeneIntC::Save(std::ostream &strm) const
  {
    strm << "GeneInt " << m_value;
    return true;
  }

  //! Save to binary stream
  bool GeneIntC::Save(RavlN::XMLOStreamC &strm) const {
    GeneC::Save(strm);
    strm << XMLAttribute("value",m_value) ;
    return true;
  }

  XMLFactoryRegisterConvertC<GeneIntC,GeneC> g_registerGeneInt("RavlN::GeneticN::GeneIntC");
  RAVL_INITVIRTUALCONSTRUCTOR_NAMED(GeneIntC,"RavlN::GeneticN::GeneIntC");
  static RavlN::TypeNameC g_typePtrGeneInt(typeid(GeneIntC::RefT),"RavlN::SmartPtrC<RavlN::GeneticN::GeneIntC>");

  // ------------------------------------------------------------------

  //! Factory constructor
  GeneTypeFloatC::GeneTypeFloatC(const XMLFactoryContextC &factory)
   : GeneTypeC(factory),
     m_min(static_cast<float>(factory.AttributeReal("min",0.0))),
     m_max(static_cast<float>(factory.AttributeReal("max",100.0)))
  {}

  //! Constructor
  GeneTypeFloatC::GeneTypeFloatC(const std::string &name,float min,float max)
  : GeneTypeC(name),
    m_min(min),
    m_max(max)
  {}

  GeneTypeFloatC::GeneTypeFloatC(BinIStreamC &strm)
   : GeneTypeC(strm)
  {
    ByteT version = 0;
    strm >> version;
    if(version != 1)
      throw RavlN::ExceptionUnexpectedVersionInStreamC("GeneTypeFloatC");
    strm >> m_min >> m_max;
  }

  //! Load form a binary stream
  GeneTypeFloatC::GeneTypeFloatC(std::istream &strm)
   : GeneTypeC(strm)
  {
    RavlAssertMsg(0,"not implemented");
  }

  //! Save to binary stream
  bool GeneTypeFloatC::Save(BinOStreamC &strm) const
  {
    if(!GeneTypeC::Save(strm))
      return false;
    ByteT version = 1;
    strm << version << m_min << m_max;
    return true;
  }


  //! Save to binary stream
  bool GeneTypeFloatC::Save(std::ostream &strm) const
  {
    return false;
  }

  //! Generate a new value
  void GeneTypeFloatC::RandomValue(float &newValue) const {
    newValue = static_cast<float>(RavlN::Random1() * (m_max - m_min)) + static_cast<float>(m_min);
    if(newValue < m_min)
       newValue = m_min;
     if(newValue > m_max)
       newValue = m_max;
  }

  //! Create randomise value
  void GeneTypeFloatC::Random(GeneC::RefT &newGene) const
  {
    float newValue;
    RandomValue(newValue);
    newGene = new GeneFloatC(*this,newValue);
  }

  //! Mutate a gene
  bool GeneTypeFloatC::Mutate(float fraction,const GeneC &original,RavlN::SmartPtrC<GeneC> &newGene) const
  {
    if(fraction <= 0) {
      newGene = &original;
      return false;
    }
    const GeneFloatC &originalFloat = dynamic_cast<const GeneFloatC &>(original);
    float value;
    RandomValue(value);
    float newValue = fraction * static_cast<float>(originalFloat.Value()) + (1.0f - fraction) * value;
    // Clip to valid range just in case of rounding problems
    if(newValue < m_min)
      newValue = m_min;
    if(newValue > m_max)
      newValue = m_max;
    newGene = new GeneFloatC(*this,newValue);
    return true;
  }

  //!  Default values for basic types
  const GeneTypeC &GeneType(float value) {
    static GeneTypeC::RefT defaultFloatType = new GeneTypeFloatC("DefaultFloat",0.0,1.0);
    return *defaultFloatType;
  }



  XMLFactoryRegisterConvertC<GeneTypeFloatC,GeneTypeC> g_registerGeneTypeFloat("RavlN::GeneticN::GeneTypeFloatC");
  RAVL_INITVIRTUALCONSTRUCTOR_NAMED(GeneTypeFloatC,"RavlN::GeneticN::GeneTypeFloatC");
  static RavlN::TypeNameC g_typePtrGeneTypeFloat(typeid(GeneTypeFloatC::RefT),"RavlN::SmartPtrC<RavlN::GeneticN::GeneTypeFloatC>");

  // ------------------------------------------------------------------

  //! Factory constructor
  GeneFloatC::GeneFloatC(const XMLFactoryContextC &factory)
   : GeneC(factory),
     m_value(static_cast<float>(factory.AttributeReal("value",0.0)))
  {
    RavlAssertMsg(dynamic_cast<GeneTypeFloatC *>(m_type.BodyPtr()) != 0,"Wrong type");
  }

  //! Constructor
  GeneFloatC::GeneFloatC(const GeneTypeC &geneType,float value)
   : GeneC(geneType),
     m_value(value)
  {}

  //! Load form a binary stream
  GeneFloatC::GeneFloatC(BinIStreamC &strm)
   : GeneC(strm)
  {
    ByteT version = 0;
    strm >> version;
    if(version != 1)
      throw RavlN::ExceptionUnexpectedVersionInStreamC("GeneC");
    strm >> m_value;
  }

  //! Load form a binary stream
  GeneFloatC::GeneFloatC(std::istream &strm)
   : GeneC(strm)
  {
    RavlAssertMsg(0,"not implemented");
  }

  //! Save to binary stream
  bool GeneFloatC::Save(BinOStreamC &strm) const
  {
    if(!GeneC::Save(strm))
      return false;
    ByteT version = 1;
    strm << version << m_value;
    return true;
  }


  //! Save to binary stream
  bool GeneFloatC::Save(std::ostream &strm) const
  {
    strm << "GeneInt " << m_value;
    return false;
  }

  //! Save to binary stream
  bool GeneFloatC::Save(RavlN::XMLOStreamC &strm) const {
    GeneC::Save(strm);
    strm << XMLAttribute("value",m_value) ;
    return true;
  }


  XMLFactoryRegisterConvertC<GeneFloatC,GeneC> g_registerGeneFloat("RavlN::GeneticN::GeneFloatC");
  RAVL_INITVIRTUALCONSTRUCTOR_NAMED(GeneFloatC,"RavlN::GeneticN::GeneFloatC");
  static RavlN::TypeNameC g_typePtrGeneFloat(typeid(GeneFloatC::RefT),"RavlN::SmartPtrC<RavlN::GeneticN::GeneFloatC>");

}}

