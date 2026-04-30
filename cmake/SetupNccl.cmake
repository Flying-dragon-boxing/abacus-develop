include_guard(GLOBAL)

function(abacus_setup_nccl target_name)
  find_library(NCCL_LIBRARY NAMES nccl
      HINTS ${NCCL_PATH} ${NVHPC_ROOT_DIR}
      PATH_SUFFIXES lib lib64 comm_libs/nccl/lib)
  find_path(NCCL_INCLUDE_DIR NAMES nccl.h
      HINTS ${NCCL_PATH} ${NVHPC_ROOT_DIR}
      PATHS ${CUDAToolkit_ROOT}
      PATH_SUFFIXES include comm_libs/nccl/include)

  if(NOT NCCL_LIBRARY OR NOT NCCL_INCLUDE_DIR)
    message(FATAL_ERROR
      "NCCL not found. Set NCCL_PATH or NVHPC_ROOT_DIR.")
  endif()

  message(STATUS "Found NCCL for parallel_device: ${NCCL_LIBRARY}")
  if(NOT TARGET NCCL::NCCL)
    add_library(NCCL::NCCL IMPORTED INTERFACE)
    set_target_properties(NCCL::NCCL PROPERTIES
        INTERFACE_LINK_LIBRARIES "${NCCL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}")
  endif()

  include_directories(${NCCL_INCLUDE_DIR})
  target_link_libraries(${target_name} NCCL::NCCL)
endfunction()
