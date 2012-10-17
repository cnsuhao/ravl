/*! author="Charles Galambos" */

#include "ccmath/ccmath.h"
#include <iostream>

using namespace std;

int main()
{
  int data[32];
  for(int i = 3;i < 200000;i++) {
    int n = pfac(i,data,'o');
    std::cerr << i << " : " << n << " - ";
    int total = 1;
    for(int j = 1;j <= data[0];j++) {
      total *= data[j];
      std::cerr << " " << data[j];
    }
    if(total != n) 
      std::cerr << " ** Failed ** ";
    std::cerr << " = " << total << "\n";
  }
  return 0;
}
