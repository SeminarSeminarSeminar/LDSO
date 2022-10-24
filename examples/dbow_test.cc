#include <csignal>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <sys/time.h>

#include "frontend/FullSystem.h"
#include "DatasetReader.h"


float setting_desiredImmatureDensity = 1500;

int main(int argc, char* argv[]){
	// Load Vocabulary
	ORBVocabulary orb3_vocabulary;
	orb3_vocabulary.load("./vocab/orbvoc.dbow3");

	// Load DBow Database
	DBoW3::QueryResults results;
	DBow3::DBoW3Database keyframe_database;
	DBow3::BowVector frame_bow;
	DBow3::FeatureVector feature_vector;

	// load image

	vector<Feature> features;
	vector<cv::Mat> descriptors;
	vector<int> bow_id;


	FeatureDetector detector;
	features.reserve(setting_desiredImmatureDensity);
	detector.DetectCorners(stting_desiredImmatureDensity, frame);
	
	for(auto &feature: features){
		feature->ip = shared_ptr<ImmaturePoint>(
			new ImmaturePoint(frame, feature, 1 Hcalib->mpCH);
		)
	}

	for(const auto &feature: features){
		if(feature->isCorner) {
			cv::Mat m(1,32, CV_8U);
			for (int k = 0; k<32; k++){
				m.data[k] = feature.descriptor[k];
			}
			descriptors.push_back(m);
			bow_id.push_back(i);
		}
	}
	orb3_vocabulary->transform(descriptors, frame_bow, feature_vector, 4);
	
	keyframe_database.query(frame->bowVec, results, 1, 0);


	return 0;
}
