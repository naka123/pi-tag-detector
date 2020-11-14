#ifndef __IPA_FIDUCIAL_MODEL_PI_H__
#define __IPA_FIDUCIAL_MODEL_PI_H__

#include "FiducialDefines.h"
#include "AbstractFiducialModel.h"

#include "FiducialPiParameters.h"


namespace ipa_Fiducials
{

/// Struct to represent a pi fiducial
struct t_pi
{
	void sparse_copy_to(t_pi& copy)
	{
		copy.parameters.m_id = parameters.m_id;
		copy.marker_points = marker_points;
	}

	FiducialPiParameters parameters;

	double cross_ration_0; ///< Cross ration for line type 0
	double cross_ration_1; ///< Cross ration for line type 1

	std::vector<std::vector<cv::RotatedRect> > fitting_image_lines_0; ///< lines that fit to the first cross ratio 
	std::vector<std::vector<cv::RotatedRect> > fitting_image_lines_1; ///< lines that fit to the second cross ratio

	int no_matching_lines; ///< Number of matching sides (at most 4 per marker)

	std::vector<cv::Point2f> marker_points; ///< ellipse coordinates in marker coordinate system
	std::vector<cv::RotatedRect> image_points; ///< ellipse coordinates in marker coordinate system
};


/// @class FiducialModelPi
///
/// A concrete class to represent a fiducial
class __DLL_LIBFIDUCIALS__ FiducialModelPi : public AbstractFiducialModel
{
public:

	//*******************************************************************************
	// AbstractFiducial interface implementation
	//*******************************************************************************
	FiducialModelPi();
	~FiducialModelPi();

    unsigned long GetPoints(cv::Mat& image, std::vector<t_points>& vec_points);

	/// Locates the fiducial within the image and inferes the camera pose from it
	/// @param scene image
	/// @return <code>RET_FAILED</code> if no tag could be detected
	/// <code>RET_OK</code> on success
	unsigned long GetPose(cv::Mat& image, std::vector<t_pose>& vec_pose);

	/// Load fiducial-centric coordinates of markers from file
	/// @param directory Directory, where the parameters of all fiducials are stores
	unsigned long LoadParameters(std::string directory_and_filename);
	unsigned long LoadParameters(std::vector<FiducialPiParameters> pi_tags);

	std::string GetType()
	{
		return "PI";
	};

	//*******************************************************************************
	// Class specific functions
	//*******************************************************************************

private:

	bool TagUnique(std::vector<t_pi>& tag_vec, t_pi& newTag);
	bool AnglesValid2D(std::vector<cv::RotatedRect>& image_points);
	bool ProjectionValid(cv::Mat& rot_CfromO, cv::Mat& trans_CfromO, cv::Mat& camera_matrix,
		cv::Mat& pattern_coords, cv::Mat& image_coords);

	std::vector<t_pi> m_ref_tag_vec; ///< reference tags to be recognized
	cv::Mat m_debug_img; ///< image that holds debugging output
	bool m_use_fast_pi_tag;
	bool m_debug = false;
	int m_min_ellipse_size;
	int m_max_ellipse_size;
	int m_min_matching_lines = 4;
	double m_cross_ratio_max_dist = 0.03;
};

} // end namespace ipa_Fiducials

#endif // __IPA_FIDUCIAL_MODEL_PI_H__
