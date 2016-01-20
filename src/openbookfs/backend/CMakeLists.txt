configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/config.yaml)


cc_require(fuse Boost THREADS libcrypto++ protobuf ProtobufCompiler re2 tclap 
           yaml-cpp SOCI sigc++)
if(required_FOUND)
  include_directories(${fuse_INCLUDE_DIRS}
                      ${Boost_INCLUDE_DIRS}
                      ${THREADS_INCLUDE_DIRS}
                      ${protobuf_INCLUDE_DIRS}
                      ${libcrypto++_INCLUDE_DIRS}
                      ${re2_INCLUDE_DIRS}
                      ${tclap_INCLUDE_DIRS}
                      ${yaml-cpp_INCLUDE_DIRS}
                      ${SOCI_INCLUDE_DIRS}
                      ${sigc++_INCLUDE_DIRS}
                      ${CMAKE_CURRENT_SOURCE_DIR})
    

  set(LIBS ${LIBS} 
      ${CMAKE_THREAD_LIBS_INIT}
      ${Boost_LIBRARIES}
      ${fuse_LDFLAGS}
      ${protobuf_LDFLAGS}
      ${THREADS_LIBRARIES}
      ${libcrypto++_LDFLAGS}
      ${re2_LDFLAGS}
      ${yaml-cpp_LDFLAGS}
      ${sigc++_LDFLAGS}
      ${SOCI_LIBRARY}
      ${SOCI_sqlite3_PLUGIN}
      cpp_pthreads_static)
        
  add_executable(obfs_backend 
                 main.cpp
                 fuse_operations.cpp
                 Backend.cpp
                 Connection.cpp
                 Database.cpp
                 FileContext.cpp
                 FuseContext.cpp
                 LongJob.cpp
                 MessageHandler.cpp
                 MountPoint.cpp
                 SocketListener.cpp
                 VersionVector.cpp
                 ../jobs/SendTree.cpp
                 ../jobs/SendFile.cpp
                 ../messages.cpp
                 ../FdSet.cpp
                 ../FileDescriptor.cpp
                 ../Marshall.cpp
                 ../NotifyPipe.cpp
                 ../SelectSpec.cpp
                 ../TimeVal.cpp
                 ../TimeSpec.cpp
                 ${MESSAGES_PB_CC})
  set_target_properties(obfs_backend PROPERTIES 
                        COMPILE_FLAGS "-D_FILE_OFFSET_BITS=64")
  target_link_libraries(obfs_backend ${LIBS})
else()
  mark_excluded("obfs_backend: ${required_MISSING}")
endif()



