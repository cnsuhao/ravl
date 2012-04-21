// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2011, Charles Galambos
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! lib=RavlXMLFactory
//! author="Charles Galambos"
//! docentry=Ravl.API.Core.IO.Services

#include "Ravl/ServiceList.hh"
#include "Ravl/XMLFactoryAutoStart.hh"
#include "Ravl/XMLFactory.hh"
#include "Ravl/OS/SysLog.hh"

namespace RavlN {

  //! Constructor.
  ServiceListC::ServiceListC(const XMLFactoryContextC &factory)
  {
    Load(factory);
  }

  //! Default constructor.
  ServiceListC::ServiceListC()
  {}

  //! Destructor
  ServiceListC::~ServiceListC()
  {
    Shutdown();
  }

  //! Load services from the given factory context.
  bool ServiceListC::Load(const XMLFactoryContextC &factory) {
    if(!factory.UseComponentGroup("Services",m_services)) {
      RavlWarning("No services section found under '%s'. ",factory.Path().data());
      return false;
    }
    //! Start in the order they appear in the file.
    for(unsigned i = 0;i < m_services.size();i++)
      m_services[i]->Start();
    return true;
  }

  //! Shutdown remaining services.
  bool ServiceListC::Shutdown() {
    //! Shutdown in reverse order.
    for(int i = static_cast<int>(m_services.size())-1;i >= 0;i--) {
      RavlAssert(m_services[i].IsValid());
      //RavlDebug("Shutting down %d %s ",i,m_services[i]->Name().data());
      m_services[i]->Shutdown();
    }
    m_services.clear();
    return true;
  }

  void LinkServiceList()
  {}

}
