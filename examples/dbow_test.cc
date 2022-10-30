#include <csignal>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <sys/time.h>

#include "Feature.h"
#include "Frame.h"
#include "internal/ImmaturePoint.h"


#include "frontend/FullSystem.h"
#include "DatasetReader.h"
#include "DBoW3/src/DBoW3.h"
#include "DBoW3/src/Database.h"


int main(int argc, char* argv[]){
	// Load Vocabulary
	shared_ptr<ORBVocabulary> orb3_vocabulary(new ORBVocabulary());
	orb3_vocabulary->load(argv[3]);
	std::cout << "vocabulary loaded\n";


	// Load DBow Database
	DBoW3::QueryResults results;
	DBoW3::BowVector frame_bow;
	DBoW3::FeatureVector feature_vector;
	shared_ptr<DBoW3::Database> keyframe_database(new DBoW3::Database(*orb3_vocabulary));

	keyframe_database->load(argv[4]);
	std::cout << "database loaded\n";

	// load image
	shared_ptr<ImageFolderReader> reader(new ImageFolderReader(ImageFolderReader::KITTI, argv[1], argv[2],"",""));
	shared_ptr<ImageAndExposure> img(reader->getImage(0));
	reader->setGlobalCalibration();
	std::cout << "reader set done\n";

	// Create Frame
	shared_ptr<Camera> camera(new Camera(fxG[0], fyG[0], cxG[0], cyG[0]));
	shared_ptr<Frame> frame(new Frame(img->timestamp));
	frame->CreateFH(frame);
	shared_ptr<FrameHessian> frame_hessian = frame->frameHessian;
	frame_hessian->ab_exposure = img->exposure_time;
	frame_hessian->makeImages(img->image, camera->mpCH);
	std::cout << "frame created\n";

	// Loop Detection
	FeatureDetector detector;
	frame->features.reserve(setting_desiredImmatureDensity);
	detector.DetectCorners(setting_desiredImmatureDensity, frame);
	std::cout << "features detected\n";	
	for(auto &feature: frame_hessian->frame->features){
		feature->ip = shared_ptr<ImmaturePoint>(new ImmaturePoint(frame_hessian->frame, feature, 1, camera->mpCH));
	}
	frame->ComputeBoW(orb3_vocabulary);
	std::cout << "bag of words computed\n";

	keyframe_database->query(frame->bowVec, results, 1, -1);
	if(results.empty()){
		std::cout << "no loop found\n";
	}
	DBoW3::Result r = results[0];
	std::cout << r.Id <<"\n";



	return 0;
}
