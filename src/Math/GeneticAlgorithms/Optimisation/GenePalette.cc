
#include "Ravl/Genetic/GenePalette.hh"
#include "Ravl/Genetic/GeneType.hh"

namespace RavlN { namespace GeneticN {

  //! Holds information used when mutating, crossing or generating genes.

  GenePaletteC::GenePaletteC(UInt32T seed)
   : m_random(seed)
  {
    static RCHashC<StringC,SmartPtrC<GeneTypeC> > emptyMap(true);
    m_proxyMap.Push(emptyMap);
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

  //! Push new proxy map on the stack.
  void GenePaletteC::PushProxyMap(const RCHashC<StringC,SmartPtrC<GeneTypeC> > &newMap)
  {
    m_proxyMap.Push(newMap);
  }

  //! Pop old map off the stack.
  void GenePaletteC::PopProxyMap()
  {
    m_proxyMap.DelTop();
  }

}}
