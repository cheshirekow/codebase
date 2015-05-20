configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/config.yaml)


set(cmake_depends Boost THREADS SOCI SOCI_sqlite3)
set(pkg_depends fuse libcrypto++ re2 sigc++ tclap yamlcpp)

check_found(depends ${cmake_depends} ${pkg_depends})
if(depends_FOUND)
  cc_executable(obfs_backend
                SOURCES main.cpp
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
                CMAKE_DEPENDS ${cmake_depends}
                PKG_DEPENDS ${pkg_depends}
                TARGET_DEPENDS cpp_pthreads_static)
  target_link_libraries(obfs_backend_exe ${SOCI_sqlite3_PLUGIN})
        
  set_target_properties(obfs_backend_exe PROPERTIES
                        COMPILE_FLAGS "-D_FILE_OFFSET_BITS=64")
else()
  message(WARNING "Can't build obfs_backend, missing ${depends_MISSING}")
endif()



