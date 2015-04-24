#Note(josh): This listfile was not distributed with json_spirit

if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIR})

  set(LIBRARY_HEADERS json_spirit/json_spirit_reader.h
                      json_spirit/json_spirit_value.h
                      json_spirit/json_spirit_writer.h
                      json_spirit/json_spirit.h
                      json_spirit/json_spirit_error_position.h
                      json_spirit/json_spirit_reader_template.h
                      json_spirit/json_spirit_stream_reader.h
                      json_spirit/json_spirit_utils.h
                      json_spirit/json_spirit_writer_options.h
                      json_spirit/json_spirit_writer_template.h)

  set(LIBRARY_SOURCES json_spirit/json_spirit_reader.cpp
                      json_spirit/json_spirit_value.cpp
                      json_spirit/json_spirit_writer.cpp  )

  add_library(json_spirit_static STATIC ${LIBRARY_SOURCES})
  add_library(json_spirit_shared SHARED ${LIBRARY_SOURCES})

  set_target_properties(json_spirit_static json_spirit_shared
                        PROPERTIES OUTPUT_NAME json_spirit)

  install(FILES ${LIBRARY_HEADERS}
          DESTINATION include/json_spirit)

  install(TARGETS json_spirit_static
                  json_spirit_shared
          RUNTIME DESTINATION bin
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib )
endif()

