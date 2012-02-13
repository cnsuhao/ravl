/*
 * testStringSpeed.cc
 *
 *  Created on: 4 Aug 2011
 *      Author: charles
 */

#include "Ravl/String.hh"

int main(int nargs,char **argv) {

  for(int i = 0;i < 1000000;i++) {
    RavlN::StringC theString(i);
    RavlN::StringC hello("hello");
    RavlN::StringC bye("bye");
    RavlN::StringC add;
    add += hello;
    bye += bye;
  }


  return 0;
}
