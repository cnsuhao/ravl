#include <cppunit/extensions/HelperMacros.h>

#include "Ravl/Prob/PDFContinuousDesignerNormal.hh"
#include "Ravl/Random.hh"

using namespace RavlProbN;
	
class PDFBooleanTest: public CppUnit::TestCase {
	CPPUNIT_TEST_SUITE( PDFBooleanTest );
	CPPUNIT_TEST( testCreatePDF );
	CPPUNIT_TEST_SUITE_END();
private:
    VariableContinuousC m_variable;
	PDFContinuousDesignerNormalC m_designer;
public:
	void setUp() {
		m_variable = VariableContinuousC("Normal", RealRangeC(-10.0, 10.0));
		m_designer = PDFContinuousDesignerNormalC::getInstance();
	}
	
	void tearDown() {
	}
	
	void testCreatePDF() {
		DListC<RealT> realSamples;
		for (int i = 0; i < 1000; i++) {
			realSamples.InsLast(RandomGauss());
		}
		PDFContinuousAbstractC pdf = m_designer.CreatePDF(m_variable, realSamples);
		cout << pdf.ToString() << endl;
	}
	
};

CPPUNIT_TEST_SUITE_REGISTRATION( PDFBooleanTest );
