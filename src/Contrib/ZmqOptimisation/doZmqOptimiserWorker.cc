// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2011, Charles Galambos
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! lib=RavlZmq

#include "Ravl/Zmq/Context.hh"
#include "Ravl/Zmq/Socket.hh"
#include "Ravl/Option.hh"
#include "Ravl/RLog.hh"
#include "Ravl/StrStream.hh"
#include "Ravl/XMLFactory.hh"
#include "Ravl/Resource.hh"
#include "Ravl/IO.hh"
#include "Ravl/DP/Blackboard.hh"
#include "Ravl/Point2d.hh"

using namespace RavlN;
using namespace RavlN::ZmqN;

int main(int nargs, char **argv)
{
  OptionC opts(nargs, argv);
  SetResourceRoot(opts.String("i", PROJECT_OUT, "Install location"));
  StringC
  configFile = opts.String("c", PROJECT_OUT "/share/Ravl/Zmq/exZmqGeneticOptimiser.xml", "Configuration file");
  StringC logFile = opts.String("l", "stderr", "Checkpoint log file. ");
  StringC logLevel = opts.String("ll", "info", "Logging level (debug, info, warning, error)");
  bool verbose = opts.Boolean("v", false, "Verbose mode.");
  opts.Check();

  RLogInit(nargs, argv, logFile.chars(), verbose);
  RLogSubscribeL(logLevel.chars());

  XMLFactoryC::RefT mainFactory = new RavlN::XMLFactoryC(configFile);
  XMLFactoryContextC factory(*mainFactory);

  rInfo("Opening Receiver socket");
  SocketC::RefT receiver;
  if (!factory.UseComponent("Receiver", receiver)) {
    RavlError("Failed to open Receiver socket!");
    return 1;
  }

  rInfo("Opening Sink socket");
  SocketC::RefT sink;
  if (!factory.UseComponent("Sink", sink)) {
    RavlError("Failed to open Sink socket!");
    return 1;
  }

  rInfo("Optimise Worker ready to go!");

  while (true) {

    // Get data
    SArray1dC<char> data;
    if (!receiver->Recieve(data)) {
      rError("Failed to receive optimiser data!");
      continue;
    }
    BufIStreamC bis(data);
    BinIStreamC is(bis);
    BlackboardC bb;
    is >> bb;

    IntT id = 0;
    if (!bb.Get("id", id)) {
      rError("Failed to get id");
      continue;
    }

    // Some knowledge of what is exactly in the Blackboard is actually required here!
    RealT row = 0.0;
    if (!bb.Get("row", row)) {
      rError("Failed to get row");
      continue;
    }

    RealT col = 0.0;
    if (!bb.Get("col", col)) {
      rError("Failed to get col");
      continue;
    }
    Point2dC point(row, col);

    rInfo("Evaluating task %d and location (%0.2f, %0.2f)", id, point.Row(), point.Col());

    // Do optimisation....
    Point2dC target(0.1234, 0.4321);
    RealT score = (1.0 / (point.EuclideanDistance(target) + 0.0001));

    // Send back data
    BlackboardC results(true);
    results.Put("id", id);
    results.Put("score", score);

    BufOStreamC bos;
    {
      BinOStreamC os(bos);
      os << results;
    }

    if (!sink->Send(bos.Data())) {
      rError("Trouble sending result");
      continue;
    }
    // OK ready for next task
  }

  return 0;
}
