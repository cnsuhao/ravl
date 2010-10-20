
namespace RavlN {

  extern void linkDistance();
  extern void linkDistanceCityBlock();
  extern void linkDistanceEuclidean();
  extern void linkDistanceMahalanobis();
  extern void linkDistanceMax();
  extern void linkDistanceRobust();
  extern void linkDistanceSqrEuclidean();
  extern void linkDistanceChi2();

  void LinkDistance() {
    linkDistance();
    linkDistanceCityBlock();
    linkDistanceEuclidean();
    linkDistanceMahalanobis();
    linkDistanceMax();
    linkDistanceRobust();
    linkDistanceSqrEuclidean();
    linkDistanceChi2();
  }
  
}
