# Find the NVML library
#
# The following variables are optionally searched for defaults
#  NVML_ROOT_DIR:    Base directory where all NVML components are found
#
# The following are set after configuration is done:
#  NVML_FOUND
#  NVML_INCLUDE_DIR
#  NVML_LIBRARY

file (GLOB MLPATH /usr/lib/nvidia-???)
find_path(NVML_INCLUDE_DIR NAMES nvml.h PATHS  ${CUDA_INCLUDE_DIRS} ${NVML_ROOT_DIR}/include)
find_library(NVML_LIBRARY nvidia-ml PATHS ${MLPATH} /usr/local/cuda/lib64/stubs ${NVML_ROOT_DIR}/lib ${NVML_ROOT_DIR}/lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVML DEFAULT_MSG NVML_INCLUDE_DIR NVML_LIBRARY)

if(NVML_FOUND)
  message(STATUS "Found NVML (include: ${NVML_INCLUDE_DIR}, library: ${NVML_LIBRARY})")
  mark_as_advanced(NVML_INCLUDE_DIR NVML_LIBRARY)
endif()

