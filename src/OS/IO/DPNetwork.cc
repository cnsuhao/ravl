
#include "Ravl/DP/DPNetwork.hh"
#include "Ravl/SysLog.hh"
#include "Ravl/XMLFactoryRegister.hh"

namespace RavlN {

  //: Default constructor.
  // Creates an empty network

  DPNetworkC::DPNetworkC()
  {}

  //: Create from an XML factory.
  DPNetworkC::DPNetworkC(const XMLFactoryContextC &factory)
  {
    XMLFactoryContextC componentContext;
    if(factory.ChildContext("Components",componentContext)) {
      m_components = CollectionC<RCAbstractC>(componentContext.Children().Size());
      for(RavlN::DLIterC<XMLTreeC> it(componentContext.Children());it;it++) {
        RCAbstractC value;
        if(!componentContext.UseChildComponent(it->Name(),value,false)) {
          RavlError("Failed to load child component %s, at %s ",it->Name().data(),componentContext.Path().data());
          throw RavlN::ExceptionBadConfigC("Failed to load component");
        }
        m_components.Append(value);

        XMLFactoryContextC ff;
        componentContext.ChildContext(it->Name(),ff);
        StringC aName;

        if(!(aName = ff.AttributeString("exportAsOPort")).IsEmpty()) {
          DPOPortBaseC aPortBase(value);
          if(!aPortBase.IsValid()) {
            RavlError("Not an OPort %s ",aName.c_str());
            throw RavlN::ExceptionBadConfigC("Not an OPort.");
          }
          m_oports.InsLast(aPortBase);
        }

        if(!(aName = ff.AttributeString("exportAsIPort")).IsEmpty()) {
          DPIPortBaseC aPortBase(value);
          if(!aPortBase.IsValid()) {
            RavlError("Not an IPort %s ",aName.c_str());
            throw RavlN::ExceptionBadConfigC("Not an IPort.");
          }
          m_iports.InsLast(aPortBase);
        }

        DPStreamOpC aStreamOp(value);
        if(aStreamOp.IsValid()) {
          if(!(aName = ff.AttributeString("exportOPlug")).IsEmpty()) {
            DPOPlugBaseC oplug;
            if(!aStreamOp.GetOPlug(aName,oplug)) {
              RavlError("Failed for find OPlug %s ",aName.c_str());
              throw RavlN::ExceptionBadConfigC("Failed to find OPlug.");
            }
            m_oplugs.InsLast(oplug);
          }
          if(!(aName = ff.AttributeString("exportIPlug")).IsEmpty()) {
            DPIPlugBaseC iplug;
            if(!aStreamOp.GetIPlug(aName,iplug)) {
              RavlError("Failed for find IPlug %s ",aName.c_str());
              throw RavlN::ExceptionBadConfigC("Failed to find OPlug.");
            }
            m_iplugs.InsLast(iplug);
          }
          if(!(aName = ff.AttributeString("exportOPort")).IsEmpty()) {
            DPOPortBaseC oport;
            if(!aStreamOp.GetOPort(aName,oport)) {
              RavlError("Failed for find OPort %s ",aName.c_str());
              throw RavlN::ExceptionBadConfigC("Failed to find OPlug.");
            }
            m_oports.InsLast(oport);
          }
          if(!(aName = ff.AttributeString("exportIPort")).IsEmpty()) {
            DPIPortBaseC iport;
            if(!aStreamOp.GetIPort(aName,iport)) {
              RavlError("Failed for find IPort %s ",aName.c_str());
              throw RavlN::ExceptionBadConfigC("Failed to find OPlug.");
            }
            m_iports.InsLast(iport);
          }
        }
      }
    }

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

  static XMLFactoryRegisterConvertC<DPNetworkC,DPStreamOpBodyC> g_registerDPNetwork("RavlN::DPNetworkC");

  void LinkDPNetwork()
  {}
}
