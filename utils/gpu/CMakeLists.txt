cuda_add_library(lit_sgncr_toolbox_gpu SHARED
        cuda_utils.h
        cuda_utils.cu
  )
set_target_properties(lit_sgncr_toolbox_gpu PROPERTIES VERSION ${GADGETRON_VERSION_STRING} SOVERSION ${GADGETRON_SOVERSION})

target_link_libraries(lit_sgncr_toolbox_gpu Gadgetron::gadgetron_core Gadgetron::gadgetron_toolbox_cpucore_math Gadgetron::gadgetron_toolbox_spiral Gadgetron::gadgetron_toolbox_cpunfft)

target_include_directories(lit_sgncr_toolbox_gpu
        PUBLIC    
		$<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/utils/>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/non_cartesian/>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/waveforms/>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/spiral/>
)

target_link_libraries(lit_sgncr_toolbox_gpu
  gadgetron_toolbox_hostutils 
  gadgetron_toolbox_mri_core
  gadgetron_toolbox_gpusdc
  gadgetron_toolbox_cpusdc
  gadgetron_toolbox_gpucore
  gadgetron_toolbox_gpusolvers
  gadgetron_toolbox_gpuoperators
	gadgetron_toolbox_gpuparallelmri
	gadgetron_mri_noncartesian
  gadgetron_toolbox_spiral 
  gadgetron_toolbox_cpunfft
  gadgetron_toolbox_gpunfft
  gadgetron_toolbox_log
  ${CUDA_LIBRARIES}
  Boost::program_options
  )
 

install(TARGETS lit_sgncr_toolbox_gpu
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	RUNTIME DESTINATION bin
	COMPONENT main
)

install(FILES
  cuda_utils.h
  DESTINATION ${GADGETRON_INSTALL_INCLUDE_PATH}
  COMPONENT main)