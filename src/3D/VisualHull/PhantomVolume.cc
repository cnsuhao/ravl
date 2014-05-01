// Ravl includes
//#include "Ravl/.hh"

#if 0
#include "../Matte/SilhouetteImage.hh"
using namespace MVTMatteN;
#include "../../MaxFlow/graph.hh"
#endif

#include "Ravl/VisualHull/VisualHull.hh"
#include "Ravl/VisualHull/VoxelClean.hh"
#include "Ravl/VisualHull/VoxelDistanceTransform.hh"
#include "Ravl/VisualHull/PhantomVolume.hh"

namespace MVTHullN
{

   // Wrapper to delete phantom volumes from a voxel grid
   void DeletePhantomVolumes(
      ByteVoxelGridC& voxels, 
      const PinholeCameraArrayC& cameras, 
      const SArray1dC<ImageC<ByteT> >& silimages, 
      IntT erosionDepth, 
      IntT dilateDepth,
      bool bSingleRegion,
      bool verbose)
   {
      PhantomVolumeC phantoms(voxels, cameras);
      phantoms.ExtractPhantoms(silimages, erosionDepth, dilateDepth, bSingleRegion, verbose);
   }

   // Build the layered depth representation for the voxel set
   void
      PhantomVolumeC::BuildDepthLayers(
      HashC<Index3dC, IntT>& depthLayers,
      Array3dC<ByteT>& work,
      const ByteVoxelGridC& voxels,
      const PinholeCameraArrayC& cameras,
      const SArray1dC<ImageC<ByteT> >& silimages,
      IntT depth)
   {
      depthLayers.Empty();

      // Compute the distance transform to the voxel surface
      VoxelDistanceTransformC dt;
      dt.InitSurfaceVoxels(work, voxels);
      dt.ChamferDistance(work);

      // Now define the depth layers in the voxel set up to the given depth
      ByteT thresh = voxels.OccupiedThreshold();
      Array3dIter2C<ByteT, ByteT> ita(work, voxels.Array());
      IndexRange3dIterC iti(work.Frame());
      for (; ita; ita++, iti++)
      {
         if (ita.Data2()>thresh && ita.Data1()<=depth)
         {
            depthLayers[ iti.Data() ] = ita.Data1();
         }
      }
   }

   // Derive the silhouette constraints for a silhouette image
   DListC< DListC<Index3dC> > 
      PhantomVolumeC::SilhouetteConstraints(
      const ImageC<ByteT>& silimage,
      const ByteVoxelGridC& voxels,
      const PinholeCameraC& camera)
   {
      DListC< DListC<Index3dC> > ret;

      // Derive the set of contour pixels
      HSetC<Index2dC> silcontour = ExtractSilhouetteContour(silimage);
      // Determine the valid voxel projection for the contour pixels
      HashC<Index2dC, DListC<Index3dC> > proj = VoxelProjection(silcontour, voxels, camera);

      // Store the intersection constraints
      HSetIterC<Index2dC> itf(silcontour);
      for (; itf; itf++)
      {
         Index2dC pix = itf.Data();
         // Iterate over the voxels f or this feature pixel
         if (proj.IsElm(pix))
         {
            ret += proj[itf.Data()];
         }
      }

      return ret;
   }

   // Perform a graph cut in the voxel set to extract the surface
   // NOTE: a 26-connected neighbourhood is used to provide a much smoother final voxel surface
   // an 18-connected neighbourhood tends to result in pock-marks
   // Constraints that are inconsistent with the silhouette constraints are deleted from constraints 
   void 
      PhantomVolumeC::ExtractSurface(
      ByteVoxelGridC& voxels,
      const SArray1dC< DListC< DListC<Index3dC> > >& silconstraints,
      const HashC<Index3dC, IntT >& depthLayers,
      bool verbose,
      IntT numlayers,
      FloatT ballooninc0,
      RealT k,
      IntT nsizeLimit)
   {
      IntT maxdepth = numlayers-1;
      Array3dC<ByteT>& arr = voxels.Array();

      // Iteratively construct and cut the graph until all silhouette constraints are satisfied
      HSetC<IntT> satisfied; // Do not recheck satisfied silhouette constraints
      HashC<Index3dC, FloatT> balloonterm;
      FloatT ballooninc = ballooninc0;
      IntT previnvalidconstraints = 0;
      IntT invalidconstraints = 0;
      do
      {
         if (verbose)
         {
            cerr << "  constructing graph...\n";
         }

         // Create the graph structure
         Graph *g = new Graph();

         // Create a node for every voxel
         HashC< Index3dC, Graph::node_id > nlookup;
         HashIterC<Index3dC, IntT > itd(depthLayers);
         for (; itd; itd++)
         {
            const Index3dC& vox = itd.Key();
            IntT l = itd.Data();
            Graph::node_id n = g->add_node();
            nlookup[ vox ] = n;
            Graph::captype slink = 0, tlink = 0;
            // Connect the first layer and all snodes to the source
            if (l==0)
            {
               slink = Graph::InfiniteCap();
            }
            else if (l==maxdepth)
            {
               tlink = Graph::InfiniteCap();
            }
            else 
            {
               // Add any required ballooning term
               FloatT b;
               if (balloonterm.Lookup(vox,b))
               {
                  tlink += Graph::Cap(b);
               }
            }
            if (slink != 0 || tlink != 0)
            {
               g -> set_tweights(n, slink, tlink);
            }
         }

         // Create the data and neighbourhood edges for every node
         HashIterC< Index3dC, Graph::node_id > itn(nlookup);
         HashC< Graph::node_id, DListC<Graph::node_id> > dlinks;
         for (; itn; itn++)
         {
            const Index3dC& vox = itn.Key();
            Graph::node_id n = itn.Data();
            IntT l = depthLayers[vox];
            // Find the minimum neighbourhood size to connect to a deeper layer
            IntT nsize = 1;
            if (l < maxdepth)
            {
               for (; nsize <= nsizeLimit; nsize++)
               {
                  bool valid = false;
                  IndexRange3dIterC itrng(IndexRange3dC(vox,nsize));
                  for (; itrng; itrng++)
                  {
                     const Index3dC& nn = itrng.Data();
                     if (nn != vox)
                     {
                        IntT nnl;
                        if (depthLayers.Lookup(nn, nnl))
                        {
                           if (nnl == l+1)
                           {
                              valid = true;
                              break;
                           }
                        }
                     }
                  }
                  if (valid) break;
               }
            }
            RealT score = 0.5;
            IndexRange3dIterC itrng(IndexRange3dC(vox,nsize));
            for (; itrng; itrng++)
            {
               const Index3dC& nn = itrng.Data();
               if (nn != vox)
               {
                  Graph::node_id nnodeid;
                  if (nlookup.Lookup(nn,nnodeid))
                  {
                     IntT nnl = depthLayers[nn];
                     if (nnl == l)
                     {
                        Graph::captype w = Graph::Cap(k);
                        g -> add_edge(n, nnodeid, w, w);
                     }
                     else if (nnl == l+1)
                     {
                        RealT dist = Vector3dC(vox.I() - nn.I(), vox.J() - nn.J(), vox.K() - nn.K()).Magnitude();
                        Graph::captype w = Graph::Cap(((1.0 - score) / dist));
                        g -> add_edge(n, nnodeid, w, Graph::InfiniteCap());
                        dlinks[n] += nnodeid;
                     }
                  }
               }
            }
         }

         if (verbose)
         {
            cerr << "  graph-cut...\n";
         }

         // Perform max-flow
         g -> maxflow();

         if (verbose)
         {
            cerr << "  checking silhouette constraints...";
         }

         // Determine the surface
         HSetC<Index3dC> surface;
         for (itn.First(); itn; itn++)
         {
            const Index3dC& vox = itn.Key();
            Graph::node_id n = itn.Data();
            if (g->what_segment(n) == Graph::SOURCE)
            {
               bool issurface = false;
               // Check for a sink node in the neighbourhood
               if (dlinks.IsElm(n))
               {
                  DLIterC<Graph::node_id> itdlink(dlinks[n]);
                  for (; itdlink; itdlink++)
                  {
                     if (g->what_segment(itdlink.Data())==Graph::SINK)
                     {
                        issurface = true;
                        break;
                     }
                  }
               }
               if (issurface)
               {
                  surface += vox;
               }
            }
         }

         // Check that all silhouette constraints are satisfied
         previnvalidconstraints = invalidconstraints;
         invalidconstraints = 0;
         HSetC<Index3dC> balloon;
         IntT scount = 0;
         SArray1dIterC< DListC< DListC<Index3dC> > > itsc(silconstraints);
         for (; itsc; itsc++)
         {
            DLIterC< DListC<Index3dC> > itsil(itsc.Data());
            for (; itsil; itsil++, scount++)
            {
               // Do not check satisfied constraints
               if (satisfied.Contains(scount))
               {
                  continue;
               }
               // Check that one point on this constraint is a surface point
               bool valid = false;
               DLIterC<Index3dC> it(itsil.Data());
               for (; it; it++)
               {
                  if (surface.Contains(it.Data()))
                  {
                     valid = true;
                     break;
                  }
               } // Loop over constraint points
               if (valid)
               {
                  // Remove this constraint so that we do not need to check it again
                  // i.e once satisfied always satisfied
                  satisfied += scount;
               }
               else
               {
                  // If not valid we need to increment the ballooning term for the internal region
                  invalidconstraints++;
                  for (it.First(); it; it++)
                  {
                     const Index3dC& silvox = it.Data();
                     // Check that we haven't already applied ballooning at this voxel
                     if (!balloon.Contains(silvox) && depthLayers.IsElm(silvox))
                     {
                        // Add the internal region for this point to the balloon set
                        DListC<Index3dC> queue;
                        queue += silvox;
                        balloon.Insert(silvox);
                        while (!queue.IsEmpty())
                        {
                           Index3dC vox = queue.PopFirst();
                           IntT l = depthLayers[vox];
                           // Get the neighbouring points on increasing layers
                           IntT ii,jj,kk;
                           for (ii=-1; ii<2 ; ii++)
                           {
                              for (jj=-1; jj<2 ; jj++)
                              {
                                 for (kk=-1; kk<2 ; kk++)
                                 {
                                    if ((ii==0 && jj==0 && kk==0))                                 
                                    {
                                       continue;
                                    }
                                    Index3dC nn(vox.I()+ii,vox.J()+jj,vox.K()+kk);
                                    if (nlookup.IsElm(nn) && depthLayers[nn]==l+1)
                                    {
                                       if (!balloon.Contains(nn))
                                       {
                                          queue += nn;
                                          balloon.Insert(nn);
                                       }
                                    }
                                 }
                              }
                           }
                        }
                     }
                  } // Add internal region for each constraint point
               } // If not valid balloon the internal region
            } // Check silhouette constraints
         }

         if (verbose)
         {
            cerr << invalidconstraints << " invalid\n";
         }
         if (invalidconstraints != 0)
         {
            // Increment the ballooning term for internal region
            HSetIterC<Index3dC> it(balloon);
            for (; it; it++)
            {
               const Index3dC& vox = it.Data();
               if (balloonterm.IsElm(vox))
               {
                  balloonterm[vox] += ballooninc;
               }
               else
               {
                  balloonterm[vox] = ballooninc;
               }
            }
         }

         // Erode the voxel set down to the set of source side nodes adjacent to sink nodes
         if (invalidconstraints == 0 || invalidconstraints == previnvalidconstraints)
         {
            for (itn.First(); itn; itn++)
            {
               const Index3dC& vox = itn.Key();
               Graph::node_id n = itn.Data();
               if (g->what_segment(n) == Graph::SOURCE && !surface.Contains(vox))
               {
                  arr[vox] = 0;
               }
               else
               {
                  arr[vox] = 255;
               }
            }
         }

         delete g;
         ballooninc += ballooninc0;
      } while (invalidconstraints != 0 && invalidconstraints != previnvalidconstraints);
   }

   // Extract the phantom volumes in a visual-hull reconstruction
   // Returns the voxel grid array filled with the phantom volumes
   Array3dC<ByteT>
      PhantomVolumeC::ExtractPhantoms(
      const SArray1dC<ImageC<ByteT> >& silimages, 
      IntT erosionDepth, 
      IntT dilateDepth,
      bool bSingleRegion,
      bool verbose)
   {
      Array3dC<ByteT> ret(m_voxels.Array().Frame());

      if (verbose)
      {
         cerr << "  Deriving silhouette constraints...\n";
      }
      SArray1dC< DListC< DListC<Index3dC> > > silconstraints(m_cameras.Size());
      UIntT cam;
      for (cam = 0; cam < m_cameras.Size(); cam++)
      {
         DListC< DListC<Index3dC> > camsilconstraints = SilhouetteConstraints(silimages[cam], m_voxels, m_cameras[cam]);
         silconstraints[cam] = camsilconstraints;
      }

      if (verbose)
      {
         cerr << "  Building depth layers...\n";
      }
      // Build the depth layers inside the visual-hull
      HashC<Index3dC, IntT> depthLayers;
      BuildDepthLayers(depthLayers, ret, m_voxels, m_cameras, silimages, erosionDepth);

      if (verbose)
      {
         cerr << "  Extracting surface...\n";
      }
      // Cache the visual hull in the working array
      Array3dIter2C<ByteT, ByteT> it(ret, m_voxels.Array());
      for (; it; it++)
      {
         it.Data1() = it.Data2();
      }
      // Reconstruct the minimal volume solution that satisfies the silhouette constraints
      ExtractSurface(m_voxels, silconstraints, depthLayers, verbose, erosionDepth);

      // Dilate the voxel surface to restore the original
      // This will neglect thin protruding phantom volumes that cannot be recovered in dilation
      IntT itcount = 0;
      for (; itcount<dilateDepth; itcount++)
      {
         // Derive the voxel surface
         HSetC<Index3dC> surface = ExtractSurfaceVoxels(m_voxels);
         // Dilate the surface
         Array3dIter2C<ByteT, ByteT> itarr(m_voxels.Array(), ret);
         IndexRange3dIterC itvox(ret.Frame());
         for (; itarr; itarr++, itvox++)
         {
            if (!itarr.Data1() && itarr.Data2())
            {
               const Index3dC& vox = itvox.Data();
               // Check whether this voxel is connected to the surface in a 6-connected neighbourhood
               if (surface.Contains(Index3dC(vox.I()-1,vox.J()  ,vox.K()  )) ||
                  surface.Contains(Index3dC(vox.I()+1,vox.J()  ,vox.K()  )) ||
                  surface.Contains(Index3dC(vox.I()  ,vox.J()-1,vox.K()  )) ||
                  surface.Contains(Index3dC(vox.I()  ,vox.J()+1,vox.K()  )) ||
                  surface.Contains(Index3dC(vox.I()  ,vox.J()  ,vox.K()-1)) ||
                  surface.Contains(Index3dC(vox.I()  ,vox.J()  ,vox.K()+1)))
               {
                  itarr.Data1() = itarr.Data2();
               }
            }
         }
      }

      // Remove any remaining disconnected voxels
      if (bSingleRegion)
      {
         VoxelCleanC clean(m_voxels);
         clean.ExtractSingleRegion();
      }

      // Now take the difference in the voxel sets to return the phantom volumes
      Array3dIter2C<ByteT, ByteT> itarr(m_voxels.Array(), ret);
      for (; itarr; itarr++)
      {
         if (itarr.Data1())
         {
            itarr.Data2() = 0;
         }
      }

      return ret;
   }

}
