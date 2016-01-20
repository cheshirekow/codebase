set( OUTPUT exp_regularity )

cc_require(eigen3 GTKmm TinyXML2 yamlcpp Boost)
if(required_FOUND)
    include_directories(
        ${eigen3_INCLUDE_DIRS}
        ${GTKmm_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        )
        
    set( LIBS ${LIBS} 
         ${GTKmm_LDFLAGS}
         ${yamlcpp_LIBRARY}
         ${Boost_LIBRARIES}
         ${TinyXML2_LIBRARY} )
         
    configure_file( config.cpp ${CMAKE_CURRENT_BINARY_DIR}/config.cpp )
        
    add_executable( ${OUTPUT} #EXCLUDE_FROM_ALL
                    main.cpp 
                    ${CMAKE_CURRENT_BINARY_DIR}/config.cpp
                    )
                    
    target_link_libraries( ${OUTPUT} ${LIBS} )
                    
else()
   mark_excluded("${OUTPUT}: missing ${rquired_MISSING}")
endif()


