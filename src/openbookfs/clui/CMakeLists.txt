cc_require(Boost THREADS protobuf libcrypto++ re2 tclap yaml-cpp SOCI sigc++)
if(required_FOUND)
  include_directories(${Boost_INCLUDE_DIRS}s
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
      ${protobuf_LDFLAGS}
      ${THREADS_LIBRARIES}
      ${libcrypto++_LDFLAGS}
      ${re2_LDFLAGS}
      ${yaml-cpp_LDFLAGS}
      ${sigc++_LDFLAGS}
      ${SOCI_LIBRARY}
      ${SOCI_sqlite3_PLUGIN})
        
  add_executable(obfs
                 connection.cpp
                 main.cpp
                 Options.cpp
                 ../messages.cpp
                 ../FdSet.cpp
                 ../FileDescriptor.cpp
                 ../Marshall.cpp
                 ../NotifyPipe.cpp
                 ../SelectSpec.cpp
                 ../TimeVal.cpp
                 ${MESSAGES_PB_CC}
                 commands/Checkout.cpp
                 commands/Connect.cpp
                 commands/ListKnownPeers.cpp
                 commands/ListMounts.cpp
                 commands/LoadConfig.cpp
                 commands/Release.cpp
                 commands/SaveConfig.cpp
                 commands/SetClientSocket.cpp
                 commands/SetDisplayName.cpp
                 commands/SetLocalSocket.cpp
                 commands/SetRemoteSocket.cpp
                 commands/StartSync.cpp)
                     
  set_target_properties(obfs PROPERTIES 
                        COMPILE_FLAGS "-D_FILE_OFFSET_BITS=64")
  target_link_libraries(obfs ${LIBS})
else()
  mark_excluded("obfs: missing ${required_MISSING}")
endif()



