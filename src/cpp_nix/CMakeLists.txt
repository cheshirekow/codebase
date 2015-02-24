set(LIBRARY_SOURCES
  fd_set.cc
  notify_pipe.cc
  select_set.cc
  timespec.cc
  timeval.cc)

add_library(cpp_nix_static STATIC ${LIBRARY_SOURCES})
add_library(cpp_nix_shared SHARED ${LIBRARY_SOURCES})
set_target_properties(
  cpp_nix_static 
  cpp_nix_shared
  PROPERTIES OUTPUT_NAME cpp_nix)
  
install(TARGETS 
  cpp_nix_static 
  cpp_nix_shared
  EXPORT CPP_NIX_TARGETS
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)