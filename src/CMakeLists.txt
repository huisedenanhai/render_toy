enable_language(CUDA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /utf-8")

###############################
# find dependencies
###############################
# find CUDA
find_package(CUDA 10.0 REQUIRED)
get_filename_component(CUDA_LIBRARY_DIR ${CUDA_cudart_static_LIBRARY} DIRECTORY)

# find Optix 7
if (WIN32)
  set(OPTIX_7_DEFAULT_INCLUDE_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.1.0/include")
elseif ()
  set(OPTIX_7_DEFAULT_INCLUDE_DIR OPTIX_7_INCLUDE_DIR-NOTFOUND)
endif ()

set(OPTIX_7_INCLUDE_DIR "${OPTIX_7_DEFAULT_INCLUDE_DIR}" CACHE PATH "include directory of Optix 7")
if (NOT EXISTS "${OPTIX_7_INCLUDE_DIR}")
  message(FATAL_ERROR "OPTIX_7_INCLUDE_DIR not found: ${OPTIX_7_INCLUDE_DIR}")
endif ()

add_executable(obj2ma
  obj2ma.cpp
)

target_compile_features(obj2ma PRIVATE
  cxx_std_17
)

target_include_directories(obj2ma PRIVATE
  ${CMAKE_SOURCE_DIR}/ext
)

add_executable(toy_exe
  main.cpp
  math_consts.h
  rotation.h
  context.h
  context.cpp
  scene.h
  scene.cpp
  integrator.h
  integrator.cpp
  path_tracing.h
  path_tracing.cpp
  exceptions.h
  )
target_compile_features(toy_exe PRIVATE
  cxx_std_17
)
target_include_directories(toy_exe PRIVATE
  ${CMAKE_SOURCE_DIR}/ext
  ${CUDA_INCLUDE_DIRS}
  ${OPTIX_7_INCLUDE_DIR}
  )
target_link_directories(toy_exe PRIVATE
  ${CUDA_LIBRARY_DIR}
  )
target_link_libraries(toy_exe PRIVATE
  cuda
  cudart
  )

#######################################
# runtime loadable ptx 
#######################################
add_library(toy_ptx OBJECT 
  pipeline.cu
  )
target_compile_options(toy_ptx PRIVATE 
  --gpu-architecture=compute_30        # OptiX 4 supports only Kepler and newer. OptiX 6 only Maxwell and newer.
  --use_fast_math                      # Generate fast approximations. Requires CUDA 8.0 or newer nvrtc to work!
  --relocatable-device-code=true       # Generate PTX code also for unreferenced functions. Needed since CUDA 8.0 to generate code for callable programs.
  --generate-line-info                 # Generate line information to allow better profiling with Nsight.
  -D__x86_64                           # Generate 64-bit code only. Must be set or OptiX headers won't work.
  )
target_include_directories(toy_ptx PRIVATE
  ${OPTIX_7_INCLUDE_DIR}
  )
set_property(TARGET toy_ptx PROPERTY CUDA_PTX_COMPILATION ON)

add_custom_target(toy ALL
  COMMAND ${CMAKE_COMMAND} 
  "-DOBJECTS=$<TARGET_OBJECTS:toy_ptx>"
  "-DOUTPUT_DIR=$<TARGET_FILE_DIR:toy_exe>/ptx"
  -P ${CMAKE_CURRENT_SOURCE_DIR}/copy_objects.cmake
  COMMENT "copy ptx to exe dir"
  )
add_dependencies(toy toy_ptx)