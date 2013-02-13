
#include "Ravl/DP/DPNetwork.hh"

namespace RavlN {

  //: Default constructor.
  // Creates an empty network

  DPNetworkC::DPNetworkC()
  {}

  //: Create from an XML factory.
  DPNetworkC::DPNetworkC(const XMLFactoryContextC &factory)
  {
    factory.UseComponentGroup("Components",m_components);
    for(unsigned i = 0;i < m_components.Size();i++) {

    }
#if 0
    factory.UseComponentGroup("IPlugs",m_iplugs);
    factory.UseComponentGroup("OPlugs",m_oplugs);
    factory.UseComponentGroup("IPorts",m_iports);
    factory.UseComponentGroup("OPorts",m_oports);
#endif
    Setup(factory);
  }

  DListC<DPIPlugBaseC> DPNetworkC::IPlugs() const {
    return m_iplugs;
  }
  //: Input plugs.

  DListC<DPOPlugBaseC> DPNetworkC::OPlugs() const {
    return m_oplugs;
  }
  //: Output plugs

  DListC<DPIPortBaseC> DPNetworkC::IPorts() const {
    return m_iports;
  }
  //: Input ports.

  DListC<DPOPortBaseC> DPNetworkC::OPorts() const {
    return m_oports;
  }
  //: Output ports

}
