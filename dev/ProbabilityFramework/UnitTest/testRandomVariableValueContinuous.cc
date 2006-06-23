#include <cppunit/extensions/HelperMacros.h>

#include "Ravl/Prob/RandomVariableValueContinuous.hh"

using namespace RavlProbN;
	
class RandomVariableValueContinuousTest: public CppUnit::TestCase {
	CPPUNIT_TEST_SUITE( RandomVariableValueContinuousTest );
	CPPUNIT_TEST_EXCEPTION( testCreateThrows, ExceptionC );
	CPPUNIT_TEST( testToString );
	CPPUNIT_TEST( testValue );
	CPPUNIT_TEST( testEquality );
	CPPUNIT_TEST_SUITE_END();
private:
	RandomVariableContinuousC m_variable;
	RandomVariableValueContinuousC m_value0, m_value0_5, m_value1;
public:
	void setUp() {
		m_variable = RandomVariableContinuousC("variable", RealRangeC(0.0, 1.0));
		m_value0 = RandomVariableValueContinuousC(m_variable, 0.0);
		m_value0_5 = RandomVariableValueContinuousC(m_variable, 0.5);
		m_value1 = RandomVariableValueContinuousC(m_variable, 1.0);
	}
	
	void tearDown() {
	}
	
	void testCreateThrows() {
		RandomVariableValueContinuousC v1(m_variable, -0.1);
		RandomVariableValueContinuousC v2(m_variable, 1.1);
	}
	
	void testToString() {
		CPPUNIT_ASSERT( m_value0.ToString() == "0.000000" );
		CPPUNIT_ASSERT( m_value0_5.ToString() == "0.500000" );
		CPPUNIT_ASSERT( m_value1.ToString() == "1.000000" );
	}
	
	void testValue() {
		CPPUNIT_ASSERT( m_value0.Value() == 0.0 );
		CPPUNIT_ASSERT( m_value0_5.Value() == 0.5 );
		CPPUNIT_ASSERT( m_value1.Value() == 1.0 );
	}
	
	void testEquality() {
		RandomVariableValueContinuousC secondValue0_5(m_variable, 0.5);
		CPPUNIT_ASSERT( m_value0 == m_value0 );
		CPPUNIT_ASSERT( m_value0_5 == m_value0_5 );
		CPPUNIT_ASSERT( m_value0_5 == secondValue0_5 );
		CPPUNIT_ASSERT( !(m_value0 == m_value0_5) );
		CPPUNIT_ASSERT( m_value0 != m_value0_5 );
		CPPUNIT_ASSERT( !(m_value0 != m_value0) );
		CPPUNIT_ASSERT( !(m_value0_5 != secondValue0_5) );
	}
};

CPPUNIT_TEST_SUITE_REGISTRATION( RandomVariableValueContinuousTest );
