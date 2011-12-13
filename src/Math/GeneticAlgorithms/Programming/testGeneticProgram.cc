
#include "Ravl/Genetic/GPVariable.hh"

int testVariable();

int main(int nargs,char **argv)
{
  int ln = 0;
  if((ln = testVariable()) != 0) {
    std::cerr <<"Test failed on line " << ln << "\n";
    return 1;
  }
  return 0;
}


int testVariable()
{
  GPVariableC<int> var1;
  GPVariableC<int> var2;


  return 0;
}
