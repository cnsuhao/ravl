#include <cppunit/extensions/HelperMacros.h>

#include "Ravl/Prob/RandomVariableBoolean.hh"

using namespace RavlProbN;
	
class RandomVariableBooleanTest: public CppUnit::TestCase {
	CPPUNIT_TEST_SUITE( RandomVariableBooleanTest );
	CPPUNIT_TEST( testValue );
	CPPUNIT_TEST_SUITE_END();
private:
	RandomVariableBooleanC m_variable;
public:
	void setUp() {
		m_variable = RandomVariableBooleanC("variable");
	}
	
	void tearDown() {
	}
	
	void testValue() {
		CPPUNIT_ASSERT( m_variable.Value(true) == "variable" );
		CPPUNIT_ASSERT( m_variable.Value(false) == "¬variable" );
	}
};

CPPUNIT_TEST_SUITE_REGISTRATION( RandomVariableBooleanTest );
