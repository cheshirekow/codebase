find_package( Eigen3 )
find_package( OGRE COMPONENTS RenderSystem_GL )
find_package( OIS )

if( (Eigen3_FOUND) AND (OGRE_FOUND) )

    include_directories(
        ${Eigen3_INCLUDE_DIR}
        ${OGRE_RenderSystem_GL_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/TutorialFramework
        ${OIS_INCLUDE_DIRS}
        ${OGRE_INCLUDE_DIRS} 
        ${OGRE_RenderSystem_GL_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        )
        
    set( LIBS ${LIBS} 
        ${OIS_LIBRARIES}
        ${OGRE_LIBRARIES}
        ${OGRE_RenderSystem_GL_LIBRARIES}
        ${OGRE_RenderSystem_GL_LIBRARY} )
    
    add_executable( mpblocks_convex_hull_demo3d
                    main.cpp
                    TutorialFramework/BaseApplication.cpp 
                    )
                    
   target_link_libraries( mpblocks_convex_hull_demo3d
                            ${LIBS} )

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/plugins.cfg 
                    ${CMAKE_CURRENT_BINARY_DIR}/plugins.cfg COPYONLY)
    
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/resources.cfg 
                    ${CMAKE_CURRENT_BINARY_DIR}/resources.cfg COPYONLY)
                    
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ogre.cfg 
                    ${CMAKE_CURRENT_BINARY_DIR}/ogre.cfg COPYONLY)

                    
else()
    message( WARNING "The following packages were not found, disabling "
                     "build of demo3d" )

    if( NOT (Eigen3_FOUND ) )
        message( WARNING "Eigen3" )
    endif()
    
    if( NOT (GTKmm_FOUND ) )
        message( WARNING "Ogre" )
    endif()
endif()


