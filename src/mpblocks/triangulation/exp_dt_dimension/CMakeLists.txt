set( OUTPUT exp_dt_dimension )

if((eigen3_FOUND)
    AND (GTKmm_FOUND)
    AND (yamlcpp_FOUND)
    AND (TinyXML2_FOUND)
    AND (Boost_FOUND))
    
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
    message( WARNING
        "disabling build of ${OUTPUT}, missing: ${FINDALL_${OUTPUT}_MISSING}")
endif()


