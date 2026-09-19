// Separate extension for initialization: reuse geometry helpers without
// changing a single instruction in the installed hybrid rendering kernels.
#include "hybrid_cuda.cu"
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

namespace foamsurf {
namespace {
__global__ void quantile_forward(HybridLaunchSettings s,
    const float *p, const float *density, const uint32_t *adj, const uint32_t *off,
    const float *rays, const uint32_t *starts, const float *q, uint32_t nq,
    float *depth, int32_t *owner) {
    uint32_t r=blockIdx.x*blockDim.x+threadIdx.x;
    if(r>=s.num_rays) return;
    Vec3 o=make_vec3(rays+6*r), d=normalized(make_vec3(rays+6*r+3));
    uint32_t cell=starts[r], qi=0;
    float t=s.near_plane,T=1.0f;
    for(uint32_t k=0;k<s.max_intersections && t<s.far_plane && T>s.transmittance_threshold;++k){
        float end; uint32_t next;
        bool boundary=next_boundary(cell,o,d,p,adj,off,end,next);
        bool done=!boundary || end>=s.far_plane;
        if(done) end=s.far_plane;
        else if(end<=t){cell=next;continue;}
        float sigma=fmaxf(density[cell],0.0f);
        float alpha=1.0f-expf(-sigma*(end-t));
        float Tnext=T*(1.0f-alpha);
        while(qi<nq && Tnext<q[r*nq+qi]){
            depth[r*nq+qi]=t+logf(T/q[r*nq+qi])/sigma;
            owner[r*nq+qi]=static_cast<int32_t>(cell);
            ++qi;
        }
        T=Tnext;t=end;
        if(done || qi==nq) break;
        cell=next;
    }
}

__device__ void boundary_grad(uint32_t a,uint32_t b,float c,Vec3 o,Vec3 d,
                             const float *p,float *pg){
    if(a==UINT32_MAX || b==UINT32_MAX || c==0.0f) return;
    Vec3 pa=make_vec3(p+3*a),pb=make_vec3(p+3*b);
    atomic_add_vec3(pg+3*a,cell_intersection_gradient(pa,pb,o,d)*c);
    atomic_add_vec3(pg+3*b,cell_intersection_gradient(pb,pa,o,d)*c);
}

// Differentiate d_q = t_owner + (-log(q)-sum(previous sigma*length))/sigma_owner.
// Gradients include EVERY preceding segment, including its moving boundaries.
__global__ void quantile_backward(HybridLaunchSettings s,
    const float *p,const float *density,const uint32_t *adj,const uint32_t *off,
    const float *rays,const uint32_t *starts,uint32_t nq,const float *depth,
    const int32_t *owner,const float *up,float *pg,float *dg){
    uint32_t rq=blockIdx.x*blockDim.x+threadIdx.x;
    if(rq>=s.num_rays*nq || owner[rq]<0 || up[rq]==0.0f) return;
    uint32_t r=rq/nq,target=owner[rq],cell=starts[r],previous=UINT32_MAX;
    Vec3 o=make_vec3(rays+6*r),d=normalized(make_vec3(rays+6*r+3));
    float t=s.near_plane,g=up[rq],inv=1.0f/density[target];
    for(uint32_t k=0;k<s.max_intersections && t<s.far_plane;++k){
        float end;uint32_t next;
        bool boundary=next_boundary(cell,o,d,p,adj,off,end,next);
        bool done=!boundary || end>=s.far_plane;
        if(done) end=s.far_plane;
        else if(end<=t){cell=next;continue;}
        if(cell==target){
            atomicAdd(dg+cell,-g*(depth[rq]-t)*inv);
            boundary_grad(previous,cell,g,o,d,p,pg);
            break;
        }
        float sigma=fmaxf(density[cell],0.0f);
        if(density[cell]>=0) atomicAdd(dg+cell,-g*(end-t)*inv);
        float c=-g*sigma*inv;
        boundary_grad(previous,cell,-c,o,d,p,pg);
        if(!done) boundary_grad(cell,next,c,o,d,p,pg);
        if(done) break;
        previous=cell;cell=next;t=end;
    }
}
}

using Tensors=std::vector<torch::Tensor>;
HybridLaunchSettings validate(Tensors &v,float near,float far,float threshold,int steps){
    TORCH_CHECK(v.size()==7,"expected points,density,adjacency,offsets,rays,starts,quantiles");
    for(auto &t:v){TORCH_CHECK(t.is_cuda() && t.device()==v[0].device(),"device mismatch");t=t.contiguous();}
    for(int i:{0,1,4,6}) TORCH_CHECK(v[i].scalar_type()==torch::kFloat32,"float32 required");
    for(int i:{2,3,5}) TORCH_CHECK(v[i].scalar_type()==torch::kUInt32,"uint32 topology required");
    TORCH_CHECK(v[0].dim()==2 && v[0].size(1)==3 && v[0].size(0)>0,"invalid points");
    TORCH_CHECK(v[1].numel()==v[0].size(0) && v[3].numel()==v[0].size(0)+1,"invalid density/offsets");
    TORCH_CHECK(v[4].dim()==2 && v[4].size(1)==6 && v[5].numel()==v[4].size(0),"invalid rays/starts");
    TORCH_CHECK(v[6].dim()==2 && v[6].size(0)==v[4].size(0) && v[6].size(1)>0,"invalid quantiles");
    TORCH_CHECK(far>near && near>=0 && threshold>=0 && threshold<1 && steps>0,"invalid settings");
    return {(uint32_t)v[0].size(0),0,(uint32_t)v[4].size(0),0,(uint32_t)steps,near,far,threshold};
}

std::vector<torch::Tensor> forward(Tensors v,float near,float far,float threshold,int steps){
    auto s=validate(v,near,far,threshold,steps);
    const c10::cuda::CUDAGuard guard(v[0].device());
    auto depth=torch::full_like(v[6],-1);
    auto owner=torch::full(v[6].sizes(),-1,v[6].options().dtype(torch::kInt32));
    if(s.num_rays){
        quantile_forward<<<(s.num_rays+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(s,
            v[0].data_ptr<float>(),v[1].data_ptr<float>(),v[2].data_ptr<uint32_t>(),
            v[3].data_ptr<uint32_t>(),v[4].data_ptr<float>(),v[5].data_ptr<uint32_t>(),
            v[6].data_ptr<float>(),v[6].size(1),depth.data_ptr<float>(),owner.data_ptr<int32_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {depth,owner};
}

std::vector<torch::Tensor> backward(Tensors v,torch::Tensor depth,torch::Tensor owner,
    torch::Tensor up,float near,float far,float threshold,int steps){
    auto s=validate(v,near,far,threshold,steps);
    const c10::cuda::CUDAGuard guard(v[0].device());
    TORCH_CHECK(depth.sizes()==v[6].sizes() && owner.sizes()==depth.sizes() && up.sizes()==depth.sizes(),"gradient shape mismatch");
    TORCH_CHECK(depth.device()==v[0].device() && owner.device()==v[0].device() && up.device()==v[0].device(),"gradient device mismatch");
    TORCH_CHECK(depth.scalar_type()==torch::kFloat32 && up.scalar_type()==torch::kFloat32 && owner.scalar_type()==torch::kInt32,"gradient dtype mismatch");
    depth=depth.contiguous();owner=owner.contiguous();up=up.contiguous();
    auto pg=torch::zeros_like(v[0]),dg=torch::zeros_like(v[1]);
    uint32_t n=s.num_rays*v[6].size(1);
    if(n){
        quantile_backward<<<(n+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(s,
            v[0].data_ptr<float>(),v[1].data_ptr<float>(),v[2].data_ptr<uint32_t>(),
            v[3].data_ptr<uint32_t>(),v[4].data_ptr<float>(),v[5].data_ptr<uint32_t>(),
            v[6].size(1),depth.data_ptr<float>(),owner.data_ptr<int32_t>(),up.data_ptr<float>(),
            pg.data_ptr<float>(),dg.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {pg,dg};
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",foamsurf::forward);m.def("backward",foamsurf::backward);}
