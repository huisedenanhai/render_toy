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
  set(OPTIX_7_DEFAULT_INCLUDE_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.0.0/include")
elseif ()
  set(OPTIX_7_DEFAULT_INCLUDE_DIR OPTIX_7_INCLUDE_DIR-NOTFOUND)
endif ()

set(OPTIX_7_INCLUDE_DIR "${OPTIX_7_DEFAULT_INCLUDE_DIR}" CACHE PATH "include directory of Optix 7")
if (NOT EXISTS "${OPTIX_7_INCLUDE_DIR}")
  message(FATAL_ERROR "OPTIX_7_INCLUDE_DIR not found: ${OPTIX_7_INCLUDE_DIR}")
endif ()

# find boost
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_RELEASE_LIBS ON)
find_package(Boost 1.70.0 REQUIRED COMPONENTS filesystem)

add_executable(toy_exe
  main.cpp
  exceptions.h
  )
target_include_directories(toy_exe PRIVATE
  ext
  res
  ${CUDA_INCLUDE_DIRS}
  ${OPTIX_7_INCLUDE_DIR}
  ${Boost_INCLUDE_DIR}
  )
target_link_directories(toy_exe PRIVATE
  ${CUDA_LIBRARY_DIR}
  ${Boost_LIBRARY_DIR_RELEASE}
  )
target_link_libraries(toy_exe PRIVATE
  cuda
  cudart
  nvrtc
  Boost::filesystem
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

option(ENABLE_LIVE_PREVIEW "live preview rendering process with opengl window" ON)
if (${ENABLE_LIVE_PREVIEW})
  target_compile_definitions(toy_exe PRIVATE
    -DENABLE_LIVE_PREVIEW
    )
  #######################################
  # other dependencies for live preview
  #######################################
  # find opengl
  find_package(OpenGL REQUIRED)

  target_include_directories(toy_exe PRIVATE
    ${OPENGL_INCLUDE_DIR}
    )
  target_link_libraries(toy_exe PRIVATE
    OpenGL::GL
    )

  # find GLFW
  set(GLFW_INCLUDE_DIR "GLFW_INCLUDE_DIR-NOTFOUND" CACHE PATH "include directory of glfw")
  set(GLFW_STATIC_LIB "GLFW_STATIC_LIB-NOTFOUND" CACHE FILEPATH "glfw static library")
  if (NOT EXISTS ${GLFW_INCLUDE_DIR})
    message(FATAL_ERROR "GLFW_INCLUDE_DIR ${GLFW_INCLUDE_DIR} not exist")
  endif ()
  if (NOT EXISTS ${GLFW_STATIC_LIB})
    message(FATAL_ERROR "GLFW_STATIC_LIB ${GLFW_STATIC_LIB} not exist")
  endif ()

  target_include_directories(toy_exe PRIVATE
    ${GLFW_INCLUDE_DIR}
    )
  target_link_libraries(toy_exe PRIVATE
    ${GLFW_STATIC_LIB}
    )
endif ()

add_subdirectory(ext)
