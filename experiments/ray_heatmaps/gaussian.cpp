#include <cmath>
#include <algorithm>
// CPU-only diagnostic. Planar ellipse support, not the renderer's octagon.
extern "C" void trace(int nr,int ng,int bins,double length,const double* origins,
 const double* rays,const double* g,double* tau){
 #pragma omp parallel for num_threads(4)
 for(int r=0;r<nr;r++){
  const double* o=origins+3*r;const double* d=rays+3*r;
  for(int j=0;j<ng;j++){
   const double* p=g+14*j; // mean, u/scale, v/scale, normal, opacity, kappa
   double denom=d[0]*p[9]+d[1]*p[10]+d[2]*p[11];
   if(std::abs(denom)<1e-10)continue;
   double t=((p[0]-o[0])*p[9]+(p[1]-o[1])*p[10]+(p[2]-o[2])*p[11])/denom;
   if(t<0||t>=length)continue;
   double x=o[0]+t*d[0]-p[0],y=o[1]+t*d[1]-p[1],z=o[2]+t*d[2]-p[2];
   double u=x*p[3]+y*p[4]+z*p[5],v=x*p[6]+y*p[7]+z*p[8];
   double r2=u*u+v*v;if(r2>11.3449)continue;
   double a=p[12]*std::exp(-.5*std::pow(std::max(r2,1e-8),p[13])/p[13]);
   a=std::min(a,1.-1e-12);
   tau[r*bins+int(t/length*bins)]+=-std::log1p(-a);
  }
 }
}
