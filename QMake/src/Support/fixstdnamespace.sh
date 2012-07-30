#/bin/sh
perl -pi -e 's/,istream/,std::istream/g' *.cc *.hh
perl -pi -e 's/,ostream/,std::ostream/g' *.cc *.hh
perl -pi -e 's/ ostream/ std::ostream/g' *.cc *.hh
perl -pi -e 's/ istream/ std::istream/g' *.cc *.hh
perl -pi -e 's/\(ostream/\(std::ostream/g' *.cc *.hh
perl -pi -e 's/\(istream/\(std::istream/g' *.cc *.hh
perl -pi -e 's/ endl;/ std::endl;/g' *.cc *.hh
perl -pi -e 's/type_info/std::type_info/g' *.cc *.hh
perl -pi -e 's/ cerr/ std::cerr/g' *.cc *.hh
perl -pi -e 's/ cout/ std::cout/g' *.cc *.hh
perl -pi -e 's/std::std::/std::/g' *.cc *.hh

