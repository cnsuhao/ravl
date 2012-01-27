
#include "Ravl/Genetic/GenePalette.hh"
#include "Ravl/Genetic/GeneType.hh"
#include "Ravl/XMLFactoryRegister.hh"

namespace RavlN { namespace GeneticN {

  //! Default constructor
  GeneTypeProxyMapC::GeneTypeProxyMapC()
  {}

  //! factory constructor
  GeneTypeProxyMapC::GeneTypeProxyMapC(const XMLFactoryContextC &factory)
  {

  }

  //! Load from text stream.
  GeneTypeProxyMapC::GeneTypeProxyMapC(std::istream &strm)
   : RCBodyVC(strm)
  {
    RavlAssertMsg(0,"not implemented");
  }

  //! Save to binary stream
  bool GeneTypeProxyMapC::Save(BinOStreamC &strm) const
  {
    RCBodyVC::Save(strm);
    RavlAssertMsg(0,"not implemented");
    return true;
  }

  //! Save to text stream
  bool GeneTypeProxyMapC::Save(std::ostream &strm) const {
    RCBodyVC::Save(strm);
    RavlAssertMsg(0,"not implemented");
    return true;
  }

  //! Add a new proxy to the map.
  void GeneTypeProxyMapC::AddProxy(const StringC &value,const GeneTypeC &geneType)
  {
    m_values[value] = &geneType;
  }

  //! Lookup a value.
  bool GeneTypeProxyMapC::Lookup(const StringC &key,SmartPtrC<const GeneTypeC> &val) const
  {
    return m_values.Lookup(key,val);
  }

  static XMLFactoryRegisterC<GeneTypeProxyMapC> g_registerGeneTypeProxyMap("RavlN::GeneticN::GeneTypeProxyMapC");

  // --------------------------------------------------------------------------------

  //! factory constructor
  GenePaletteC::GenePaletteC(const XMLFactoryContextC &factory)
   : m_random(factory.AttributeUInt("seed",0))
  {

  }

  //! Holds information used when mutating, crossing or generating genes.

  GenePaletteC::GenePaletteC(UInt32T seed)
   : m_random(seed)
  {
    static GeneTypeProxyMapC::RefT emptyMap = new GeneTypeProxyMapC();
    m_proxyMap.Push(*emptyMap);
  }

  //! Load from binary stream.
  GenePaletteC::GenePaletteC(BinIStreamC &strm)
   : RCBodyVC(strm)
  {
    RavlAssertMsg(0,"not implemented");
  }

  //! Load from text stream.
  GenePaletteC::GenePaletteC(std::istream &strm)
   : RCBodyVC(strm)
  {
    RavlAssertMsg(0,"not implemented");
  }

  //! Save to binary stream
  bool GenePaletteC::Save(BinOStreamC &strm) const
  {
    RCBodyVC::Save(strm);
    RavlAssertMsg(0,"not implemented");
    return true;
  }

  //! Save to text stream
  bool GenePaletteC::Save(std::ostream &strm) const {
    RCBodyVC::Save(strm);
    RavlAssertMsg(0,"not implemented");
    return true;
  }


  //! Generate a random value between 0 and 1.
  RealT GenePaletteC::RandomDouble()
  {
    return m_random.Double();
  }

  //! Generate a random integer.
  UInt32T GenePaletteC::RandomUInt32()
  {
    return m_random.UInt();
  }

  //! Generate an integer with a gaussian distribution.
  float GenePaletteC::RandomGauss()
  {
    return m_guass.Generate(m_random);
  }

  //! Add a new proxy to the map.
  void GenePaletteC::AddProxy(const StringC &value,const GeneTypeC &geneType)
  {
    GeneTypeProxyMapC::RefT newProxy = new GeneTypeProxyMapC(*m_proxyMap.Top());
    newProxy->AddProxy(value,geneType);
    m_proxyMap.Push(newProxy);
  }

  //! Push new proxy map on the stack.
  void GenePaletteC::PushProxyMap(const GeneTypeProxyMapC &newMap)
  {
    m_proxyMap.Push(&newMap);
  }

  //! Pop old map off the stack.
  void GenePaletteC::PopProxyMap()
  {
    m_proxyMap.DelTop();
  }

  static XMLFactoryRegisterC<GenePaletteC> g_registerGenePalette("RavlN::GeneticN::GenePaletteC");

}}
