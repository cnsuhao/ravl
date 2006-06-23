#include <cppunit/extensions/HelperMacros.h>

#include "Ravl/Prob/RandomVariableValueBoolean.hh"

using namespace RavlProbN;
	
class RandomVariableValueBooleanTest: public CppUnit::TestCase {
	CPPUNIT_TEST_SUITE( RandomVariableValueBooleanTest );
	CPPUNIT_TEST( testBooleanValue );
	CPPUNIT_TEST( testToString );
	CPPUNIT_TEST_SUITE_END();
private:
	RandomVariableBooleanC m_variable;
	RandomVariableValueBooleanC m_valueTrue;
	RandomVariableValueBooleanC m_valueFalse;
public:
	void setUp() {
		m_variable = RandomVariableBooleanC("variable");
		m_valueTrue = RandomVariableValueBooleanC(m_variable, true);
		m_valueFalse = RandomVariableValueBooleanC(m_variable, false);
	}
	
	void tearDown() {
	}
	
	void testBooleanValue() {
		CPPUNIT_ASSERT( m_valueTrue.BooleanValue() == true );
		CPPUNIT_ASSERT( m_valueFalse.BooleanValue() == false );
	}
	
	void testToString() {
		CPPUNIT_ASSERT( m_valueTrue.ToString() == "variable" );
		CPPUNIT_ASSERT( m_valueFalse.ToString() == "¬variable" );
	}
};

CPPUNIT_TEST_SUITE_REGISTRATION( RandomVariableValueBooleanTest );
