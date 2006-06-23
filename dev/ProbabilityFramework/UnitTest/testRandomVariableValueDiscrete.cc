#include <cppunit/extensions/HelperMacros.h>

#include "Ravl/Prob/RandomVariableValueDiscrete.hh"

using namespace RavlProbN;
	
class RandomVariableValueDiscreteTest: public CppUnit::TestCase {
	CPPUNIT_TEST_SUITE( RandomVariableValueDiscreteTest );
	CPPUNIT_TEST( testStringValue );
	CPPUNIT_TEST( testToString );
	CPPUNIT_TEST_SUITE_END();
private:
	RandomVariableDiscreteC m_variable;
	RandomVariableValueDiscreteC m_valueA;
	RandomVariableValueDiscreteC m_valueB;
public:
	void setUp() {
		HSetC<StringC> names;
		names.Insert("a");
		names.Insert("b");
		names.Insert("c");
		m_variable = RandomVariableDiscreteC("variable", names);
		m_valueA = RandomVariableValueDiscreteC(m_variable, "a");
		m_valueB = RandomVariableValueDiscreteC(m_variable, "b");
	}
	
	void tearDown() {
	}
	
	void testStringValue() {
		CPPUNIT_ASSERT( m_valueA.Value() == "a" );
		CPPUNIT_ASSERT( m_valueB.Value() == "b" );
	}
	
	void testToString() {
		CPPUNIT_ASSERT( m_valueA.ToString() == "a" );
		CPPUNIT_ASSERT( m_valueB.ToString() == "b" );
	}
};

CPPUNIT_TEST_SUITE_REGISTRATION( RandomVariableValueDiscreteTest );
