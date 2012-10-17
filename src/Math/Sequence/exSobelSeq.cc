
#include "amma/SobolSeq.hh"

int main()
{
  int k;
  for(SobolSeqC ss(1);ss.IsElm() && k < 10;ss.Next(),k++) {
    std::cerr << ss.Data()[0] << " ";
  }
  std::cerr << "\n";
  return 0;
}
