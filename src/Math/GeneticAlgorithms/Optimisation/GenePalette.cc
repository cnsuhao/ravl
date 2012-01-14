
#include "Ravl/Genetic/GenePalette.hh"
#include "Ravl/Genetic/GeneType.hh"

namespace RavlN { namespace GeneticN {

  //! Default constructor
  GeneTypeProxyMapC::GeneTypeProxyMapC()
  {}

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

  // --------------------------------------------------------------------------------

  //! Holds information used when mutating, crossing or generating genes.

  GenePaletteC::GenePaletteC(UInt32T seed)
   : m_random(seed)
  {
    static GeneTypeProxyMapC::RefT emptyMap = new GeneTypeProxyMapC();
    m_proxyMap.Push(*emptyMap);
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

}}
