// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2011, Charles Galambos
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_GENETIC_GENEPALETTE_HH
#define RAVL_GENETIC_GENEPALETTE_HH 1

#include "Ravl/RandomMersenneTwister.hh"
#include "Ravl/RandomGauss.hh"
#include "Ravl/Stack.hh"
#include "Ravl/RCHash.hh"
#include "Ravl/String.hh"
#include "Ravl/SmartPtr.hh"

namespace RavlN {
  class XMLFactoryContextC;
}

namespace RavlN { namespace GeneticN {

  class GeneTypeC;

  //! Holds information used when mutating, crossing or generating genes.

  class GenePaletteC
  {
  public:
    //! Construct from a seed.
    GenePaletteC(UInt32T seed = 0);

    //! Generate a random value between 0 and 1.
    RealT RandomDouble();

    //! Generate a random value between 0 and 1.
    RealT Random1()
    { return RandomDouble(); }

    //! Generate a random integer.
    UInt32T RandomUInt32();

    //! Generate an integer with a Gaussian distribution.
    float RandomGauss();

    //! Push new proxy map on the stack.
    void PushProxyMap(const RCHashC<StringC,SmartPtrC<GeneTypeC> > &newMap);

    //! Pop old map off the stack.
    void PopProxyMap();

    //! Access the current map
    const RCHashC<StringC,SmartPtrC<GeneTypeC> > &ProxyMap() const
    { return m_proxyMap.Top(); }

protected:
    RandomMersenneTwisterC m_random;
    RandomGaussC m_guass;
    StackC<RCHashC<StringC,SmartPtrC<GeneTypeC> > > m_proxyMap;
  };

}}

#endif
