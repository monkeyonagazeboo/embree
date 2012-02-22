// ======================================================================== //
// Copyright 2009-2011 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#define _CRT_RAND_S  

#include "integrators/pathtraceintegrator.h"
#include "renderer/materials/obj.h"
#include <typeinfo>

namespace embree
{ 
  PathTraceIntegrator::PathTraceIntegrator(const Parms& parms)
    : lightSampleID(-1), firstScatterSampleID(-1), firstScatterTypeSampleID(-1)
  {
    maxDepth        = parms.getInt  ("maxDepth"       ,10    );
    minContribution = parms.getFloat("minContribution",0.01f );
    epsilon         = parms.getFloat("epsilon"        ,128.0f)*float(ulp);
    backplate       = parms.getImage("backplate");
  }

  float mis ( float pdf1, float pdf2 )
    {
      return pdf1 / (pdf1 + pdf2);
    }
    
    __forceinline Col3f BRDF_sample(const Ref<BackendScene>& scene, Col3f c, DifferentialGeometry dg,const Sample3f wi, float epsilon)
    {
      if (wi.pdf <= 0.0f)
        return zero;
      
      Ray r( dg.P, wi.value, dg.error*epsilon, inf );
      DifferentialGeometry diff;
      scene->accel->intersect( r, diff );
      scene->postIntersect( r, diff );

      Col3f radiance =  Col3f( 0.0f,0.0f,0.0f );
      float pdflight = 0.0f;
      if ( diff.light  )
      {
	      radiance = diff.light->Le(diff, -wi.value );
	      pdflight = diff.light->pdf( dg, wi );
      }
      float weight = mis(wi.pdf, pdflight);
      if ( dot( diff.Ng, -r.dir ) > 0 )
      return radiance * c * weight / wi.pdf;
    }
    
    __forceinline Col3f LS_sample (const Ref<BackendScene>& scene,const Sample3f wo, DifferentialGeometry dg,const Sample3f wi, BRDFType giBRDFTypes, CompositedBRDF brdfs,
    Sampler* sampler, int lightSampleID)
    {
      return zero;
    }

  void PathTraceIntegrator::requestSamples(Ref<SamplerFactory>& samplerFactory, const Ref<BackendScene>& scene)
  {
    precomputedLightSampleID.resize(scene->allLights.size());

    lightSampleID = samplerFactory->request2D();
    for (size_t i=0; i<scene->allLights.size(); i++) {
      precomputedLightSampleID[i] = -1;
      if (scene->allLights[i]->precompute())
        precomputedLightSampleID[i] = samplerFactory->requestLightSample(lightSampleID, scene->allLights[i]);
    }
    firstScatterSampleID = samplerFactory->request2D((int)maxDepth);
    firstScatterTypeSampleID = samplerFactory->request1D((int)maxDepth);
  }

  Col3f PathTraceIntegrator::Li(const LightPath& lightPathOrig, const Ref<BackendScene>& scene, Sampler* sampler, size_t& numRays)
  {
	  bool done = false;
	  Col3f coeff = Col3f(1,1,1);
	  Col3f Lsum = zero;
	  Col3f L = zero;
	  LightPath lightPath = lightPathOrig;
    bool doneDiffuse = false;

    /*! while cycle instead of the recusrion call 
    * throughput is accumulated and the resulting light addition is 
    * multipliled by this throughput (coef) at each itteration */
	  while (!done)
	  {

    BRDFType directLightingBRDFTypes = (BRDFType)(DIFFUSE);
    BRDFType giBRDFTypes = (BRDFType)(ALL);

    /*! Terminate path if too long or contribution too low. */
    L = zero;

    /*! Terminate the path if maxDepth is reached */
    if (lightPath.depth >= maxDepth) // || reduce_max(coeff) < minContribution)
      return Lsum;

    /*! Traverse ray. */
    DifferentialGeometry dg;
    scene->accel->intersect(lightPath.lastRay,dg);
    scene->postIntersect(lightPath.lastRay,dg);
    const Vec3f wo = -lightPath.lastRay.dir;
    numRays++;

    /*! Environment shading when nothing hit. */
    if (!dg)
    {
      if (backplate && lightPath.unbend) {
        Vec2f raster = sampler->getPrimary();
        int width = sampler->getImageSize().x;
        int height = sampler->getImageSize().y;
        int x = (int)((raster.x / width) * backplate->width);
        x = clamp(x, 0, int(backplate->width)-1);
        int y = (int)((raster.y / height) * backplate->height);
        y = clamp(y, 0, int(backplate->height)-1);
        L = backplate->get(x, y);
      }
      else {
        if (!lightPath.ignoreVisibleLights)
          for (size_t i=0; i<scene->envLights.size(); i++)
            L += scene->envLights[i]->Le(wo);
      }
      return Lsum + L*coeff;
    }

    /*! Shade surface. */
    CompositedBRDF brdfs;
    if (dg.material) dg.material->shade(lightPath.lastRay, lightPath.lastMedium, dg, brdfs);

    /*! face forward normals */
    bool backfacing = false;
#if defined(__EMBREE_CONSISTENT_NORMALS__) && __EMBREE_CONSISTENT_NORMALS__ > 1
    return Col3f(abs(dg.Ns.x),abs(dg.Ns.y),abs(dg.Ns.z));
#else
    if (dot(dg.Ng, lightPath.lastRay.dir) > 0) {
      backfacing = true; dg.Ng = -dg.Ng; dg.Ns = -dg.Ns;
    }
#endif

    /*! Sample BRDF - get the sample direction for
    * both the indirect illumination as well as for the MIS BRDF sampling */
    Col3f c; Sample3f wi;BRDFType type;
    Vec2f s  = sampler->getVec2f(firstScatterSampleID     + lightPath.depth);
    float ss = sampler->getFloat(firstScatterTypeSampleID + lightPath.depth);
    c = brdfs.sample(wo, dg, wi, type, s, ss, giBRDFTypes);

    /*! Add light emitted by hit area light source. */
    if (!lightPath.ignoreVisibleLights && dg.light && !backfacing)
      L += dg.light->Le(dg,wo);

    /*! Check if any BRDF component uses direct lighting. */
    bool useDirectLighting = false;
    for (size_t i=0; i<brdfs.size(); i++)
      useDirectLighting |= (brdfs[i]->type & directLightingBRDFTypes) != NONE;

    /*! Direct lighting. */
    if (useDirectLighting)
    {
		  std::vector<float> illumFactor;  // illumination factor for each ls
      float sum = 0;
      LightSample ls;
      float weight = 1.0f;

		  if ( wi.pdf > 0.0f )
	    {
		    Ray r( dg.P, wi.value, dg.error*epsilon, inf );
		    DifferentialGeometry diff;
		    scene->accel->intersect( r, diff );
		    scene->postIntersect( r, diff );

		    Col3f red = Col3f( 1.0f, 0.0f, 0.0f);
		    Col3f radiance =  Col3f( 0.0f,0.0f,0.0f );
		    float pdflight = 0.0f;
		    if ( diff.light  ) // if BRDF sampling hits the light
		    {
		      radiance = diff.light->Le(diff, -wi.value );
		      pdflight = diff.light->pdf( dg, wi );
		    }
		    weight = mis(wi.pdf, pdflight);
		    if (typeid(Obj) == typeid(*dg.material) && !(type && SPECULAR)) weight = 1.0f;
        
        if ( dot( diff.Ng, -r.dir ) > 0)	
	        L += radiance * c * weight / wi.pdf;
      }

		/*! Run through all the lightsources and sample or compute the distribution function for rnd gen */
		for (size_t i=0; i<scene->allLights.size(); i++)
      {
        /*! Either use precomputed samples for the light or sample light now. */
        if (scene->allLights[i]->precompute()) ls = sampler->getLightSample(precomputedLightSampleID[i]);
        else ls.L = scene->allLights[i]->sample(dg, ls.wi, ls.tMax, sampler->getVec2f(lightSampleID));

		    /*! Start using only one random lightsource after first Lambertian reflection
        * in case of the direct illumination MIS this heuristics is ommited */ 
		    if (true)//donedif
		    {
			    /*! run through all the lighsources and compute radiance accumulatively */
          float boo = reduce_max(ls.L)/ls.tMax;  // incomming illuminance heuristic
          sum += boo;
			    illumFactor.push_back(boo);  // illumination factor
		    }
		    else  // if all the lights are sampled - take each sample and compute the addition
		    {
			    /*! Ignore zero radiance or illumination from the back. */
			    if (ls.L == Col3f(zero) || ls.wi.pdf == 0.0f || dot(dg.Ns,Vec3f(ls.wi)) <= 0.0f) continue;

			    /*! Test for shadows. */
			    bool inShadow = scene->accel->occluded(Ray(dg.P, ls.wi, dg.error*epsilon, ls.tMax-dg.error*epsilon));
			    numRays++;
			    if (inShadow) continue;

			    /*! Evaluate BRDF. */
			    L += ls.L * brdfs.eval(wo, dg, ls.wi, directLightingBRDFTypes) * rcp(ls.wi.pdf);
		    }
      }

	  /*! After fisrt Lambertian reflection pick one random lightsource and compute contribution
    * in case of MIS active this heuristic is ommited */
	  if (true && scene->allLights.size() != 0)//donedif
	  {
		  /*! Generate the random value */
		  unsigned int RndVal;  // random value
      if (rand_s(&RndVal)) std::cout << "\nRND gen error!\n";
		  // rand_r(&RndVal);
      float rnd((float)RndVal/(float)UINT_MAX);  // compute the 0-1 rnd value
		  
		  /*! Pick the particular lightsource according the intensity-given distribution */
		  size_t i = 0; 
      float accum = illumFactor[i]/sum;  // accumulative sum
		  while (i < scene->allLights.size() && rnd > accum)  // get the lightsource index accirding the Pr
      {
			  ++i;
        accum +=illumFactor[i]/sum;
      }

		  /*! Sample the selected lightsource and compute contribution */
		  if ( i >= scene->allLights.size() ) i = scene->allLights.size() -1;
      // if (usedLight != NULL)
      // std::cout << "direct light " << scene->allLights[i].ptr << "\n";
        float ql = illumFactor[i]/sum;  // Pr of given lightsource
		    // LightSample ls;
		    if (scene->allLights[i]->precompute()) ls = sampler->getLightSample(precomputedLightSampleID[i]);
		    else ls.L = scene->allLights[i]->sample(dg, ls.wi, ls.tMax, sampler->getVec2f(lightSampleID));

		    /*! Ignore zero radiance or illumination from the back. */
		    if (ls.L != Col3f(zero) && ls.wi.pdf != 0.0f && dot(dg.Ns,Vec3f(ls.wi)) > 0.0f) 
		    {
			    /*! Test for shadows. */
			    bool inShadow = scene->accel->occluded(Ray(dg.P, ls.wi, dg.error*epsilon, ls.tMax-dg.error*epsilon));
			    numRays++;
			    if (!inShadow) 
			    {
				    weight = mis( ls.wi.pdf, brdfs.pdf( wo, dg, wi, giBRDFTypes ) );
				    /*! Evaluate BRDF. */
				    L += ls.L * brdfs.eval(wo, dg, ls.wi, directLightingBRDFTypes) * rcp(ls.wi.pdf*ql) * weight;
			    }
        }
	  }
    }
	
	/* Add the resulting light */
	Lsum += coeff * L;

    /*! Global illumination. Pick one BRDF component and sample it. */
    if (lightPath.depth < maxDepth) //always true
    {
      /*! Continue only if we hit something valid. */
      if (c != Col3f(zero) && wi.pdf > 0.0f)
      {
        /*! detect the first diffuse */
        if (wi.pdf < 0.33) doneDiffuse = true;

        /*! Compute  simple volumetric effect. */
        const Col3f& transmission = lightPath.lastMedium.transmission;
        if (transmission != Col3f(one)) c *= pow(transmission,dg.t);

        /*! Tracking medium if we hit a medium interface. */
        Medium nextMedium = lightPath.lastMedium;
        if (type & TRANSMISSION) nextMedium = dg.material->nextMedium(lightPath.lastMedium);

        /*! Continue the path. */
		    float q = 1;
        if (doneDiffuse) { //std::cout << "\ndifusni\n";
          q = min(abs(reduce_max(c) * rcp(wi.pdf)), (float)1);
          // std::cout << q << "\n";
          unsigned int RndVal;
          if (rand_s(&RndVal)) std::cout << "\nRND gen error!\n";
          // rand_r(&RndVal);
          if ((float)RndVal/(float)UINT_MAX > q) { // std::cout << "konec";
            return Lsum;// + L*coeff;
          }
        }
        /*! Continue the path */
		    lightPath = lightPath.extended(Ray(dg.P, wi, dg.error*epsilon, inf), nextMedium, c, (type & directLightingBRDFTypes) != NONE);
		    coeff = coeff * c * rcp(q * wi.pdf);
      }else done = true;   // end the path
    }
  }
  return Lsum;
  }

  Col3f PathTraceIntegrator::Li(const Ray& ray, const Ref<BackendScene>& scene, Sampler* sampler, size_t& numRays) {
    return Li(LightPath(ray),scene,sampler,numRays);
  }
}