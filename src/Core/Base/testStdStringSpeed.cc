/*
 * testStringSpeed.cc
 *
 *  Created on: 4 Aug 2011
 *      Author: charles
 */

#include "Ravl/String.hh"
#include <string>
#include <iostream>
#include <sstream>

int main(int nargs,char **argv) {

  for(int i = 0;i < 1000000;i++) {
    std::stringstream ss;
    std::string s;
    ss << i;
    std::string theString = ss.str();


    std::string hello("hello");
    std::string bye("bye");
    std::string add;
    add += hello;
    bye += bye;
  }


  return 0;
}
