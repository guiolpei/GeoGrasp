#include <iostream>
#include <ctime>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/random_sample.h>

#include <pcl/ModelCoefficients.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <geograsp/GeoGrasp.h>
#include <geograsp/GraspConfigMsg.h>

const std::string GRASP_CONFIG_TOPIC = "/geograsp/grasp_config";
const int SHADOW_GRIP_TIP = 25; //25; // In mm

pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Cloud viewer"));
ros::Publisher pub;

void saveData(const pcl::PointCloud<pcl::PointNormal>::Ptr cloudFirst, 
    const pcl::PointCloud<pcl::PointNormal>::Ptr cloudSecond, 
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudObject,
    const GraspConfiguration & bestGrasp) {
  std::time_t epoch = std::time(nullptr);
  std::ostringstream converter;
  converter << epoch;

  std::string folderName = converter.str();
  mkdir(folderName.c_str(), 0777);

  pcl::io::savePCDFileBinary(folderName + "/cloud-first.pcd", *cloudFirst);
  pcl::io::savePCDFileBinary(folderName + "/cloud-second.pcd", *cloudSecond);
  pcl::io::savePCDFileBinary(folderName + "/cloud-object.pcd", *cloudObject);

  std::ofstream outFile;
  outFile.open(folderName + "/best-grasp.txt", std::ios_base::app);

  outFile << bestGrasp.firstPoint.x << ",";
  outFile << bestGrasp.firstPoint.y << ",";
  outFile << bestGrasp.firstPoint.z << ",";
  outFile << bestGrasp.secondPoint.x << ",";
  outFile << bestGrasp.secondPoint.y << ",";
  outFile << bestGrasp.secondPoint.z;
}

// Callback function for processing 3D point clouds
void cloudCallback(const sensor_msgs::PointCloud2ConstPtr & inputCloudMsg) {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  pcl::fromROSMsg(*inputCloudMsg, *cloud);

  // Remove NaN values and make it dense
  std::vector<int> nanIndices;
  pcl::removeNaNFromPointCloud(*cloud, *cloud, nanIndices);

  // Remove everything out of the working space (table)
  pcl::PassThrough<pcl::PointXYZRGB> ptFilter;
  ptFilter.setInputCloud(cloud);
  ptFilter.setFilterFieldName("z");
  ptFilter.setFilterLimits(0.0, 1.5);
  ptFilter.filter(*cloud);

  ptFilter.setInputCloud(cloud);
  ptFilter.setFilterFieldName("y");
  ptFilter.setFilterLimits(-0.55, 0.40);
  ptFilter.filter(*cloud);

  ptFilter.setInputCloud(cloud);
  ptFilter.setFilterFieldName("x");
  ptFilter.setFilterLimits(-0.50, 0.30); //(-0.70, 0.30);
  ptFilter.filter(*cloud);

  // Create the segmentation object for the planar model and set all the parameters
  pcl::SACSegmentation<pcl::PointXYZRGB> sacSegmentator;
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudPlane(new pcl::PointCloud<pcl::PointXYZRGB>());

  sacSegmentator.setModelType(pcl::SACMODEL_PLANE);
  sacSegmentator.setMethodType(pcl::SAC_RANSAC);
  sacSegmentator.setMaxIterations(50);
  sacSegmentator.setDistanceThreshold(0.025);
  sacSegmentator.setInputCloud(cloud);
  sacSegmentator.segment(*inliers, *coefficients);

  // Remove the planar inliers, extract the rest
  pcl::ExtractIndices<pcl::PointXYZRGB> indExtractor;
  indExtractor.setInputCloud(cloud);
  indExtractor.setIndices(inliers);
  indExtractor.setNegative(false);

  // Get the points associated with the planar surface
  indExtractor.filter(*cloudPlane);

  // Remove the planar inliers, extract the rest
  indExtractor.setNegative(true);
  indExtractor.filter(*cloud);

  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
  tree->setInputCloud(cloud);

  std::vector<pcl::PointIndices> clusterIndices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ecExtractor;
  ecExtractor.setClusterTolerance(0.01);
  ecExtractor.setMinClusterSize(200);
  //ecExtractor.setMaxClusterSize(25000);
  ecExtractor.setSearchMethod(tree);
  ecExtractor.setInputCloud(cloud);
  ecExtractor.extract(clusterIndices);

  if (clusterIndices.empty()) {
    // Visualize the result
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> planeColor(cloudPlane, 0, 255, 0);

    viewer->removeAllPointClouds();
    viewer->removeAllShapes();

    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "Main cloud");
    viewer->addPointCloud<pcl::PointXYZRGB>(cloudPlane, planeColor, "Plane");

    viewer->spinOnce();
  }
  else {
    std::vector<pcl::PointIndices>::const_iterator it = clusterIndices.begin();
    std::vector<geograsp::GraspConfigMsg> computedGrasps;
    int objectNumber = 0;

    viewer->removeAllPointClouds();
    viewer->removeAllShapes();

    // Every cluster found is considered an object
    for (it = clusterIndices.begin(); it != clusterIndices.end(); ++it) {
      std::cout << "======================================\n";

      pcl::PointCloud<pcl::PointXYZRGB>::Ptr objectCloud(new pcl::PointCloud<pcl::PointXYZRGB>());

      for (std::vector<int>::const_iterator pit = it->indices.begin(); 
          pit != it->indices.end(); ++pit)
        objectCloud->points.push_back(cloud->points[*pit]);

      objectCloud->width = objectCloud->points.size();
      objectCloud->height = 1;
      objectCloud->is_dense = true;

      // Create and initialise GeoGrasp
      GeoGrasp geoGraspPoints;
      geoGraspPoints.setBackgroundCloud(cloudPlane);
      geoGraspPoints.setObjectCloud(objectCloud);
      geoGraspPoints.setGripTipSize(SHADOW_GRIP_TIP);

      // Calculate grasping points
      geoGraspPoints.compute();

      // Extract best pair of points
      GraspConfiguration bestGrasp = geoGraspPoints.getBestGrasp();

      pcl::ModelCoefficients objAxisCoeff = geoGraspPoints.getObjectAxisCoeff();
      std::cout << "Obj axis: " << objAxisCoeff << "\n";

      // Visualize the result
      std::string objectLabel = "";
      std::ostringstream converter;

      converter << objectNumber;
      objectLabel += converter.str();
      objectLabel += "-";

      pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(objectCloud);
      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> planeColor(cloudPlane, 
        0, 155, 0);
      pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> planeRGB(cloudPlane);

      // Visualize the object and the table
      viewer->addPointCloud<pcl::PointXYZRGB>(objectCloud, rgb, objectLabel + "Object");
      viewer->addPointCloud<pcl::PointXYZRGB>(cloudPlane, planeRGB, objectLabel + "Plane");

      // Visualize the object's axis
      viewer->addLine(objAxisCoeff, objectLabel + "Axis vector");

      // Visualize grasping points
      viewer->addSphere(bestGrasp.firstPoint, 0.01, 0, 0, 255, objectLabel + "First best grasp point");
      viewer->addSphere(bestGrasp.secondPoint, 0.01, 255, 0, 0, objectLabel + "Second best grasp point");

      // Visualize radius (normal) clouds
      pcl::PointCloud<pcl::PointNormal>::Ptr firstPointRadiusNormalCloud(new pcl::PointCloud<pcl::PointNormal>());
      *firstPointRadiusNormalCloud = geoGraspPoints.getFirstPointRadiusNormalCloud();
      pcl::PointCloud<pcl::PointNormal>::Ptr secondPointRadiusNormalCloud(new pcl::PointCloud<pcl::PointNormal>());
      *secondPointRadiusNormalCloud = geoGraspPoints.getSecondPointRadiusNormalCloud();
      
      // Visualize the curvature of their areas
      pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointNormal> 
        firstPointNormalColorHandler(firstPointRadiusNormalCloud, "curvature");
      pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointNormal>
        secondPointNormalColorHandler(secondPointRadiusNormalCloud, "curvature");
  
      viewer->addPointCloud<pcl::PointNormal>(firstPointRadiusNormalCloud, firstPointNormalColorHandler, 
                                            objectLabel + "First point normals cloud");

      viewer->addPointCloud<pcl::PointNormal>(secondPointRadiusNormalCloud, secondPointNormalColorHandler, 
                                            objectLabel + "Second point normals cloud");
      
      // Save info
      //saveData(firstPointRadiusNormalCloud, secondPointRadiusNormalCloud, objectCloud, bestGrasp);

      // Build GraspConfigMsg
      geograsp::GraspConfigMsg graspMsg;
      graspMsg.first_point_x = bestGrasp.firstPoint.x;
      graspMsg.first_point_y = bestGrasp.firstPoint.y;
      graspMsg.first_point_z = bestGrasp.firstPoint.z;
      graspMsg.second_point_x = bestGrasp.secondPoint.x;
      graspMsg.second_point_y = bestGrasp.secondPoint.y;
      graspMsg.second_point_z = bestGrasp.secondPoint.z;
      graspMsg.obj_axis_coeff_0 = objAxisCoeff.values[0];
      graspMsg.obj_axis_coeff_1 = objAxisCoeff.values[1];
      graspMsg.obj_axis_coeff_2 = objAxisCoeff.values[2];
      graspMsg.obj_axis_coeff_3 = objAxisCoeff.values[3];
      graspMsg.obj_axis_coeff_4 = objAxisCoeff.values[4];
      graspMsg.obj_axis_coeff_5 = objAxisCoeff.values[5];
      pcl::toROSMsg<pcl::PointXYZRGB>(*objectCloud, graspMsg.object_cloud);

      computedGrasps.push_back(graspMsg);

      objectNumber++;
    }
    
    // Publish the grasp configuration of the first object
    pub.publish(computedGrasps[0]);

    //viewer->spinOnce();
    while (!viewer->wasStopped())
      viewer->spinOnce(100);
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "cloud_processor");

  viewer->initCameraParameters();
  viewer->addCoordinateSystem(0.1);

  ros::NodeHandle n("~");
  std::string cloudTopic;
  
  n.getParam("topic", cloudTopic);

  ros::Subscriber sub = n.subscribe<sensor_msgs::PointCloud2>(cloudTopic, 1, cloudCallback);
  pub = n.advertise<geograsp::GraspConfigMsg>(GRASP_CONFIG_TOPIC, 1);

  ros::spin();

  return 0;
}
