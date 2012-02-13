// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2011, Charles Galambos
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! lib=RavlXMLFactory
//! author="Charles Galambos"
//! docentry=Ravl.API.Core.Services
#ifndef RAVL_SERVICELIST_HEADER
#define RAVL_SERVICELIST_HEADER 1

#include "Ravl/Service.hh"

namespace RavlN {

  //! Class for managing a list of services.
  class ServiceListC
  {
  public:
    //! Default constructor.
    ServiceListC();

    //! Constructor.
    ServiceListC(const XMLFactoryContextC &factory);

    //! Destructor
    ~ServiceListC();

    //! Load services from the given factory context.
    bool Load(const XMLFactoryContextC &factory);

    //! Shutdown remaining services.
    bool Shutdown();

  protected:
    std::vector<ServiceC::RefT> m_services;
  };

}

#endif
