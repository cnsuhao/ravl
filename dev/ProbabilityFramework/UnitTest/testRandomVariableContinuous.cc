#include <cppunit/extensions/HelperMacros.h>

#include "Ravl/Prob/RandomVariableContinuous.hh"

using namespace RavlProbN;
	
class RandomVariableContinuousTest: public CppUnit::TestCase {
	CPPUNIT_TEST_SUITE( RandomVariableContinuousTest );
	CPPUNIT_TEST( testToString );
	CPPUNIT_TEST( testInterval );
	CPPUNIT_TEST_SUITE_END();
private:
	RandomVariableContinuousC m_variable;
public:
	void setUp() {
		m_variable = RandomVariableContinuousC("variable", RealRangeC(0.0, 1.0));
	}
	
	void tearDown() {
	}
	
	void testToString() {
	    CPPUNIT_ASSERT( m_variable.ToString() == "Variable=[0.000000,1.000000]" );
	}
	
	void testInterval() {
		CPPUNIT_ASSERT( m_variable.Interval() == RealRangeC(0.0,1.0) );
	}
	
};

CPPUNIT_TEST_SUITE_REGISTRATION( RandomVariableContinuousTest );
