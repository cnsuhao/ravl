// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2011, Charles Galambos
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! lib=RavlService
//! author="Charles Galambos"
//! docentry=Ravl.API.Core.Services
#ifndef RAVL_SERVICETHREAD_HEADER
#define RAVL_SERVICETHREAD_HEADER 1

#include "Ravl/Service.hh"
#include <vector>

namespace RavlN {
  class XMLFactoryContextC;

  //! Class to allow classes to run code after loaded in a preload section.

  class ServiceThreadC
    : public ServiceC
  {
  public:
    //! Constructor
    ServiceThreadC();

    //! XMLFactory Constructor
    ServiceThreadC(const XMLFactoryContextC &factory);

    //! Write to an std::ostream
    bool Save(std::ostream &strm) const;

    //! Write to a binary stream
    // Not implemented
    bool Save(BinOStreamC &strm) const;

    //! Start service.
    virtual bool Start();

    //! Start service with owner ptr.
    bool StartOwner();

    //! Shutdown service
    virtual bool Shutdown();

    //! Run thread directly.
    virtual bool Run();

    //! Owner reference counted ptr to class
    typedef RavlN::SmartOwnerPtrC<ServiceThreadC> RefT;

    //! Callback reference counter ptr to class
    typedef RavlN::SmartCallbackPtrC<ServiceThreadC> CBRefT;

  protected:

    //! Set to true to exit.
    bool m_started;
    bool m_terminate;

    //! Called when owner handles drop to zero.
    virtual void ZeroOwners();
  };


}

#endif
