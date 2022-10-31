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
void settingsDefault(int preset) {
    printf("\n=============== PRESET Settings: ===============\n");
    if (preset == 0 || preset == 1) {
        printf("DEFAULT settings:\n"
               "- %s real-time enforcing\n"
               "- 2000 active points\n"
               "- 5-7 active frames\n"
               "- 1-6 LM iteration each KF\n"
               "- original image resolution\n", preset == 0 ? "no " : "1x");

        //playbackSpeed = (preset == 0 ? 0 : 1);
        setting_desiredImmatureDensity = 1500;
        setting_desiredPointDensity = 2000;
        setting_minFrames = 5;
        setting_maxFrames = 7;
        setting_maxOptIterations = 6;
        setting_minOptIterations = 1;

        setting_logStuff = false;
    }
}


int main(int argc, char* argv[]){
	setting_maxAffineWeight = 0.1;
	setting_photometricCalibration = 0;
	setting_affineOptModeA = 0;
	setting_affineOptModeB = 0;
	settingsDefault(0);


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
	reader->setGlobalCalibration();
	shared_ptr<ImageAndExposure> img(reader->getImage(0));

	std::cout << "reader set done\n";

	// Create Frame
	shared_ptr<Camera> camera(new Camera(fxG[0], fyG[0], cxG[0], cyG[0]));
	camera->CreateCH(camera);


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
	
	keyframe_database->add(frame->bowVec,frame->featVec);

	keyframe_database->query(frame->bowVec, results, 1);
	if(results.empty()){
		std::cout << "no loop found\n";
		exit(1);
	}
	DBoW3::Result r = results[0];
	//keyframe_database->save("test_dbow.bin");
	std::cout << r.Id << "\n";



	return 0;
}
