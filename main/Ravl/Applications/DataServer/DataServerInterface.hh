// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2005, OmniPerception Ltd
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef _DATASERVERINTERFACE_H
#define	_DATASERVERINTERFACE_H

#include "Ravl/String.hh"
#include "Ravl/Hash.hh"
#include "Ravl/Threads/Signal1.hh"
#include "Ravl/Threads/Signal2.hh"

namespace RavlN
{

//! userlevel = Develop
//: Data server interface.

class DataServerInterfaceC
{
public:
  virtual bool AddNode(const StringC& path, const StringC& nodeType, const HashC<StringC, StringC>& options) = 0;
  //: Add a node to the live data server.
  //!param: path - The virtual path to add to the tree.
  //!param: nodeType - The node type name (matching the 'NodeType' config entry).
  //!param: options - Addition options (matching the format used for node config entries).
  //!return: True on success.

  virtual bool RemoveNode(const StringC& path, bool removeFromDisk = false) = 0;
  //: Remove a node from the live data server.
  //!param: path - The virtual path to remove from the tree.
  //!param: removeFromDisk - If a corresponding real file exists, remove it from the disk once all ports have closed.
  //!return: True on success.

  virtual Signal1C<StringC>& SignalNodeRemoved() = 0;
  //: Connect to this signal to be informed when all ports to a removed node have closed.
  //!return: Signal to a string containing the node path.

  virtual Signal2C<StringC, StringC>& SignalNodeError() = 0;
  //: Connect to this signal to be informed when a port encounters an error.
  //!return: Signal to a pair of strings containg the node path and the descriptive error.

  virtual bool QueryNodeSpace(const StringC& path, Int64T& total, Int64T& used, Int64T& available) = 0;
  //: Query the space info for a node, if applicable.
  //!param: total - Returns the space allocated for footage in bytes (both free and used). -1 if not applicable.
  //!param: used - Returns the space used for stored footage in bytes. -1 if not applicable.
  //!param: available - Returns the space available for uploading footage in bytes. -1 if not applicable.
  //!return: True if the query executed successfully.
};

}

#endif	/* _DATASERVERINTERFACE_H */
