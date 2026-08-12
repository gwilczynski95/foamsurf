find_package(CUDAToolkit REQUIRED)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -D_USE_MATH_DEFINES)

list(
  APPEND
  FOAMSURF_TOPOLOGY_INCLUDES
  ${CMAKE_SOURCE_DIR}/external/submodules/eigen)

list(
  APPEND
  FOAMSURF_TOPOLOGY_CUDA_SOURCES
  aabb_tree/aabb_tree.cu
  delaunay/delaunay.cu
  delaunay/sample_initial_tets.cu
  delaunay/growth_iteration.cu
  delaunay/delete_violations.cu
  delaunay/triangulation_ops.cu)

add_library(foamsurf_topology_upstream STATIC
            ${FOAMSURF_TOPOLOGY_CUDA_SOURCES})
target_include_directories(foamsurf_topology_upstream
                           PUBLIC ${FOAMSURF_TOPOLOGY_INCLUDES})
target_link_libraries(
  foamsurf_topology_upstream PUBLIC CUDA::cudart CUDA::cuda_driver)
target_compile_options(foamsurf_topology_upstream
                       PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -O3
                               -Wall -Wno-unknown-pragmas> -O3)
