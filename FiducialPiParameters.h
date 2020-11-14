#ifndef __IPA_FIDUCIAL_PI_PARAMETERS_H__
#define __IPA_FIDUCIAL_PI_PARAMETERS_H__

#include "AbstractFiducialParameters.h"

namespace ipa_Fiducials
{

/// @class AbstractFiduciaParameters
///
/// Abstract class to represent the parameters of a fiducial
class __DLL_LIBFIDUCIALS__ FiducialPiParameters : public AbstractFiducialParameters
{
public:
	double tag_width;
	double tag_height;

	// Assert that cross_ratio(line0) > cross_ratio(line1)
	double d_line0_AB; ///< Distance of from A to B of 4pt line A-B-C-D
	double d_line0_AC; ///< Distance of from A to C of 4pt line A-B-C-D

	double d_line1_AB; ///< Distance of from A to B of 4pt line A-B-C-D
	double d_line1_AC; ///< Distance of from A to C of 4pt line A-B-C-D
};

} // end namespace ipa_Fiducials

#endif // __IPA_FIDUCIAL_PI_PARAMETERS_H__
