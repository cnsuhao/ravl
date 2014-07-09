#ifndef RAVL_VISUALHULL_PHANTOMVOLUME_HEADER
#define RAVL_VISUALHULL_PHANTOMVOLUME_HEADER 1

//
// Extracts phantom volumes from the visual-hull
//   Reconstructs the minimum volume solution that satisfies the silhouette constraints in the images
//   Performs a morphological open operation to delete the phantom volume
//   i.e erode down to the minimum solution then dilating back up
//

#include "Ravl/Hash.hh"
#include "Ravl/Image/Image.hh"
#include "Ravl/3D/PinholeCameraArray.hh"
#include "Ravl/Voxels/VoxelGrid.hh"

namespace RavlN { namespace VisualHullN  {

  using Ravl3DN::PinholeCameraArrayC;
  using Ravl3DN::PinholeCameraC;
  using RavlImageN::ImageC;
  using VoxelsN::ByteVoxelGridC;

   class PhantomVolumeC
   {
   public:
      //:-
      // CONSTRUCTION/DESTRUCTION /////////////////////////////////////////////

      PhantomVolumeC(const ByteVoxelGridC& voxels, const PinholeCameraArrayC& cameras)
         : m_voxels(voxels), m_cameras(cameras)
      { }
      //: Default constructor

   public:

      Array3dC<ByteT> ExtractPhantoms(
         const SArray1dC<ImageC<ByteT> >& silimages, 
         IntT erosionDepth, 
         IntT dilateDepth,
         bool bSingleRegion,
         bool verbose);
      // Extract the phantom volumes in a visual-hull reconstruction
      // Deletes the phantom volumes from the visual-hull
      // Returns the voxel grid array filled with the phantom volumes
      // set erosionDepth to define the volume to extract the minimum volume solution
      // set dilateDepth to dilate the volume after satisfying the silhouettes
      // set singleRegion to extract only a single region to remove spurious volumes

   protected:

      static void BuildDepthLayers(
         HashC<Index3dC, IntT>& depthLayers,
         Array3dC<ByteT>& work,
         const ByteVoxelGridC& voxels,
         const PinholeCameraArrayC& cameras,
         const SArray1dC<ImageC<ByteT> >& silimages,
         IntT depth);
      // Build the layered depth representation for the voxel set

      static DListC< DListC<Index3dC> > SilhouetteConstraints(
         const ImageC<ByteT>& silimage,
         const ByteVoxelGridC& voxels,
         const PinholeCameraC& camera);
      // Derive the silhouette constraints for a silhouette image

      static void ExtractSurface(
         ByteVoxelGridC& voxels,
         const SArray1dC< DListC< DListC<Index3dC> > >& silconstraints,
         const HashC<Index3dC, IntT >& depthLayers,
         bool verbose,
         IntT numlayers,
         FloatT ballooninc0 = 0.2,
         RealT k = 0.01,
         IntT nsizeLimit = 5);

   protected:
      ByteVoxelGridC m_voxels;
      PinholeCameraArrayC m_cameras;
   };

   void DeletePhantomVolumes(
      ByteVoxelGridC& voxels, 
      const PinholeCameraArrayC& cameras, 
      const SArray1dC<ImageC<ByteT> >& silimages, 
      IntT erosionDepth = 10, 
      IntT dilateDepth = 3,
      bool bSingleRegion = true,
      bool verbose = true);
   // Wrapper to delete phantom volumes from a voxel grid
   // The minimum area solution that satisfies the silhouette constraints is derived
   // The solution is then dilated back to the original
   // Thin protruding phantom volumes are removed as they cannot be recovered in dilation
}}

#endif 
