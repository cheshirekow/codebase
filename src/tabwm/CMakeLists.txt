set(pkgs
    cairomm
    glog
    gflags
    pangomm
    x11
    xcomposite
    xdamage
    xext
    xfixes
    xrender)

cc_include(Boost
           eigen3
           ${pkgs})

cc_executable(tabwm_composite SOURCES poc/composite.cc PKG_DEPENDS ${pkgs})
cc_executable(basic_wm
              SOURCES basic_wm/main.cpp
                      basic_wm/util.cpp
                      basic_wm/window_manager.cpp
              PKG_DEPENDS ${pkgs})

cc_executable(ripoff
              SOURCES ripoff/main.cc
                      ripoff/util.cc
                      ripoff/window_manager.cc
              PKG_DEPENDS ${pkgs})

#add_executable(tab_wm
#               window_manager.cc
#               util.cc)
#target_link_libraries(tab_wm ${LIBS})

cc_executable(tab_decorator
              SOURCES decorator.cc
                      util.cc
              PKG_DEPENDS ${pkgs})

cc_executable(echo_server
              SOURCES echo/server.cc
              PKG_DEPENDS ${pkgs})

cc_executable(echo_client
              SOURCES echo/client.cc
              PKG_DEPENDS ${pkgs})

configure_file(xephyr_test.cc ${CMAKE_CURRENT_BINARY_DIR}/xephyr_test.cc @ONLY)
cc_executable(xephyr_test SOURCES ${CMAKE_CURRENT_BINARY_DIR}/xephyr_test.cc)

cc_test(tabwm_tree_test SOURCES "tree_test.cc" PKG_DEPENDS ${pkgs})
