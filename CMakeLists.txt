cmake_minimum_required(VERSION 3.10...3.12 FATAL_ERROR)

if (${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else ()
    cmake_policy(VERSION 3.12)
endif()

find_program(CCACHE_FOUND ccache)
if (CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif (CCACHE_FOUND)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
set(GADGETRON_INSTALL_INCLUDE_PATH include/gadgetron)

# for non-container efforts..
link_directories("${CMAKE_INSTALL_PREFIX}/lib/")

project(LIT_SGNCR_TOOLBOX LANGUAGES CXX C)

find_package(Gadgetron REQUIRED)
add_definitions(-DARMA_DONT_USE_WRAPPER -DARMA_USE_CXX11 -DARMA_64BIT_WORD)
find_package(CUDA_advanced)
find_package(Boost REQUIRED)
find_package(Python3_compat REQUIRED)



# build options for OpenMP support
find_package(OpenMP)
option(USE_OPENMP "Use OpenMP" On)
if (OPENMP_FOUND)
    if (USE_OPENMP)
        message("OpenMP multithreading enabled")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        add_definitions(-DUSE_OMP)
    else ()
        message("OpenMP multithreading is supported, but disabled")
    endif ()
else ()
    message("OpenMP multithreading not supported")
endif ()


# We actually only use system and thread explicitly, but they require linking in date_time and chrono
if (WIN32)
find_package(Boost 1.65.0 COMPONENTS system date_time chrono program_options filesystem timer exception REQUIRED )
else()
find_package(Boost 1.65.0 COMPONENTS system program_options filesystem timer REQUIRED )
endif()
if (Boost_VERSION_STRING VERSION_LESS 1.67.0)
    find_package(Boost 1.65.0 COMPONENTS python3 REQUIRED)
    set(Boost_PYTHON3_TARGET Boost::python3)
    else()
    string(REGEX MATCH "^3\\.([0-9]+)\\.[0-9]+" PYTHON_MINOR_VERSION ${PYTHONLIBS_VERSION_STRING} )
    set(PYTHON_MINOR_VERSION ${CMAKE_MATCH_1})
    find_package(Boost 1.65.0 COMPONENTS "python3${PYTHON_MINOR_VERSION}" REQUIRED)
    set(Boost_PYTHON3_FOUND TRUE)
    set(Boost_PYTHON3_TARGET Boost::python3${PYTHON_MINOR_VERSION})
endif()

include_directories( ${CUDA_INCLUDE_DIRS} )
set(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-std=c++17" )
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_STANDARD 17)
set(CURAND_FLAGS "-lcurand_static -lculibos -lcudart_static -lpthread -ldl -I /usr/local/cuda/include -L /usr/local/cuda/lib64")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CURAND_FLAGS}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set( gadgetron_lit_sgncr_toolbox_header_files
	waveforms/WaveformToTrajectory.h
	spiral/SpiralBuffer.h
	utils/util_functions.h
	utils/window_filter.h
	utils/concomitantFieldCorrection/mri_concomitant_field_correction.h
)
   
set( gadgetron_lit_sgncr_toolbox_src_files
	waveforms/WaveformToTrajectory.cpp
	utils/util_functions.cpp
	utils/GatingandBinningGadget.cpp
	spiral/SpiralAccumulateFast.cpp
	spiral/WeightsEstimationGadget.cpp
	utils/window_filter.cpp
	utils/concomitantFieldCorrection/mri_concomitant_field_correction.cpp

)

if(CUDA_FOUND)
list(APPEND gadgetron_lit_sgncr_toolbox_src_files reconstruction/SenseGriddingReconGadget3D_Parallel.cpp)
endif()

set( gadgetron_lit_sgncr_toolbox_config_files
	config/spiral_nc_example.xml
 )

add_library(lit_sgncr_toolbox SHARED  
${gadgetron_lit_sgncr_toolbox_header_files} 
${gadgetron_lit_sgncr_toolbox_src_files}
${gadgetron_lit_sgncr_toolbox_config_files}
)


target_link_libraries(lit_sgncr_toolbox Gadgetron::gadgetron_core Gadgetron::gadgetron_toolbox_cpucore_math Gadgetron::gadgetron_toolbox_spiral Gadgetron::gadgetron_toolbox_cpunfft )

target_link_libraries(lit_sgncr_toolbox gadgetron_mricore
									   gadgetron_toolbox_cpufft
									   gadgetron_toolbox_mri_core
									   gadgetron_toolbox_log
									   gadgetron_toolbox_cpuoperator
									   gadgetron_toolbox_image_analyze_io
									   gadgetron_core_parallel
									   gadgetron_toolbox_gpusdc 
									   gadgetron_toolbox_cpusdc
										)

# if(CUDA_FOUND)
    target_link_libraries(lit_sgncr_toolbox
	gadgetron_toolbox_gpucore
    gadgetron_toolbox_gpusolvers
    gadgetron_toolbox_gpuoperators
	gadgetron_toolbox_gpuparallelmri
	gadgetron_mri_noncartesian
	)
# endif()

add_subdirectory(utils)
 target_link_libraries(lit_sgncr_toolbox
	lit_sgncr_toolbox_gpu -lcurand)
install(FILES ${gadgetron_lit_sgncr_toolbox_header_files} DESTINATION ${GADGETRON_INSTALL_INCLUDE_PATH} COMPONENT main)


install(TARGETS lit_sgncr_toolbox
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	RUNTIME DESTINATION bin
	COMPONENT main
)

install(FILES ${gadgetron_lit_sgncr_toolbox_config_files} DESTINATION ${GADGETRON_INSTALL_CONFIG_PATH} COMPONENT main)
