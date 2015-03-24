if( (eigen3_FOUND)
    AND (sigc++_FOUND)
    AND (GTKmm_FOUND)
    AND (cairomm_FOUND)
    AND (yamlcpp_FOUND)
    AND (TinyXML2_FOUND)
    )
    
    include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${eigen3_INCLUDE_DIRS}
        ${sigc++_INCLUDE_DIRS}
        ${GTKmm_INCLUDE_DIRS}
        ${cairomm_INCLUDE_DIRS}
        ${yamlcpp_INCLUDE_DIR}
        ${TinyXML2_INCLUDE_DIRS}
        ) 
        
    set( LIBS ${LIBS}
        ${GTKmm_LDFLAGS}
        ${sigc++_LDFLAGS}
        ${yamlcpp_LDFLAGS}
        ${TinyXML2_LDFLAGS}
        mpblocks_gtk
         )
         
    configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/config.cpp
                    ${CMAKE_CURRENT_BINARY_DIR}/config.cpp )
    
    add_executable( mpblocks_demo_brown79 EXCLUDE_FROM_ALL
                        ${CMAKE_CURRENT_BINARY_DIR}/config.cpp
                        brown_main.cpp 
                        )
    target_link_libraries( mpblocks_demo_brown79 ${LIBS} )
else()
    message( WARNING "Skipping target demo_brown79" )
endif()

