include_directories(${Boost_INCLUDE_DIRS}
                    ${cairomm_INCLUDE_DIRS}
                    ${eigen3_INCLUDE_DIRS}
                    ${glog_INCLUDE_DIRS}
                    ${gflags_INCLUDE_DIRS}
                    ${pangomm_INCLUDE_DIRS}
                    ${x11_INCLUDE_DIRS}
                    ${xcomposite_INCLUDE_DIRS}
                    ${xdamage_INCLUDE_DIRS}
                    ${xext_INCLUDE_DIRS}
                    ${xfixes_INCLUDE_DIRS}
                    ${xrender_INDLUDE_DIRS})

set(LIBS
    ${cairomm_LDFLAGS}
    ${glog_LDFLAGS}
    ${gflags_LDFLAGS}
    ${pangomm_LDFLAGS}
    ${x11_LDFLAGS}
    ${xcomposite_LDFLAGS}
    ${xdamage_LDFLAGS}
    ${xext_LDFLAGS}
    ${xfixes_LDFLAGS}
    ${xrender_LDFLAGS})

add_executable(composite poc/composite.cc)
target_link_libraries(composite ${LIBS})

add_executable(basic_wm 
               basic_wm/main.cpp
               basic_wm/util.cpp
               basic_wm/window_manager.cpp)
target_link_libraries(basic_wm ${LIBS})

add_executable(ripoff
               ripoff/main.cc
               ripoff/util.cc
               ripoff/window_manager.cc)
target_link_libraries(ripoff ${LIBS})

#add_executable(tab_wm
#               window_manager.cc
#               util.cc)
#target_link_libraries(tab_wm ${LIBS})

add_executable(tab_decorator
               decorator.cc
               util.cc)
target_link_libraries(tab_decorator ${LIBS})

add_executable(echo_server
               echo/server.cc)
target_link_libraries(echo_server ${LIBS})

add_executable(echo_client
               echo/client.cc)
target_link_libraries(echo_client ${LIBS})
