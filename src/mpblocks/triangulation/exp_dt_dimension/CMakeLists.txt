set( OUTPUT exp_dt_dimension )

cc_require(eigen3 GTKmm TinyXML2 yamlcpp Boost)
if(required_FOUND)
    include_directories(
        ${eigen3_INCLUDE_DIR}
        ${GTKmm_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        )
        
    set( LIBS ${LIBS} 
         ${GTKmm_LDFLAGS}
         ${yamlcpp_LDFLAGS}
         ${Boost_LIBRARIES}
         ${TinyXML2_LDFLAGS} )
         
    configure_file( config.cpp ${CMAKE_CURRENT_BINARY_DIR}/config.cpp )
        
    add_executable( ${OUTPUT} EXCLUDE_FROM_ALL
                    main.cpp 
                    ExperimentBase.cpp
                    Main.cpp
                    Runner.cpp
                    ${CMAKE_CURRENT_BINARY_DIR}/config.cpp
                    )
                    
    target_link_libraries( ${OUTPUT} ${LIBS} )
                    
else()
    mark_excluded("exp_dt_dimension: missing ${required_MISSING}")
endif()


