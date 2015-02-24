find_package( Eigen3 )
find_package( MPBlocks COMPONENTS gtk cuda utility)
find_package( CUDA )

if( (Eigen3_FOUND)
    AND (MPBlocks_FOUND)
    AND (CUDA_FOUND) )

    include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${Eigen3_INCLUDE_DIR}
        ${MPBlocks_INCLUDE_DIR}
        ${CUDA_INCLUDE_DIRS}
        )
        
    set( 
        LIBS ${LIBS}
        ${MPBlocks_gtk_LIBRARY}
        ${MPBlocks_cuda_LIBRARY}
        ${MPBlocks_utility_LIBRARY}
        ${CUDA_CUDART_LIBRARY}
        ${CUDA_CUDA_LIBRARY}
        -lrt
        )
        
    cuda_add_executable( mpblocks_dubins_nearest_test
                         main.cpp
                         main.cu )
                         
    cuda_add_executable( mpblocks_dubins_nearest_plot
                         main_plot.cpp
                         main.cu )
                         
    set_source_files_properties(
            main.cpp main_plot.cpp
            PROPERTIES COMPILE_FLAGS -std=c++0x)

    target_link_libraries( mpblocks_dubins_nearest_test ${LIBS} )
    target_link_libraries( mpblocks_dubins_nearest_plot ${LIBS} )
else()
    set( MISSING "" )
    
    if( NOT (Eigen3_FOUND) )
        set( MISSING "${MISSING}, eigen3" )
    endif()
    
    if( NOT (GTKmm_FOUND) )
        set( MISSING "${MISSING}, gtkmm" )
    endif()
    
    if( NOT (Cairomm_FOUND) )
        set( MISSING "${MISSING}, cairomm" )
    endif()
    
    if( NOT (TinyXML2_FOUND) )
        set( MISSING "${MISSING}, tinyxml2" )
    endif()
    
    if( NOT (YAMLCPP_FOUND) )
        set( MISSING "${MISSING}, yaml-cpp" )
    endif()
    
    if( NOT (CUDA_FOUND) )
        set( MISSING "${MISSING}, cuda" )
    endif()
    
    message( WARNING "Missing packages, wont build dubbins curves demo. "
                     "Packages: ${MISSING}" )

endif()


    
#configure_file( 
#    ${CMAKE_CURRENT_SOURCE_DIR}/mainwindow.xml 
#    ${CMAKE_CURRENT_BINARY_DIR}/mainwindow.xml COPYONLY )
 
