# - Try to find Google Protocol Buffers compiler
# Once done, this will define
#
#  ProtobufCompiler_FOUND - system has protobuf
#  ProtobufCompiler_PATH  - protoc path

# Find compiler
set(ProtobufCompiler_FOUND "-NOTFOUND")

find_program(ProtobufCompiler_PATH
  NAMES protoc
)

if(ProtobufCompiler_PATH)
 set(ProtobufCompiler_FOUND TRUE)
endif()

if(NOT (ProtobufCompiler_FOUND) )
  message(WARNING "Failed to find google protocol buffers compiler")
else()
  message(STATUS "Found google protocol buffers compiler:\n"
                 "_PATH : ${ProtobufCompiler_PATH}\n")
endif()

# CMAKE function to run the protobuf compiler
function(protoc PROTO_FILES)
  set(OUTPUT_FILES "") 
  set(INPUT_FILES "")
  foreach(PROTO_FILE ${PROTO_FILES})
    string(REGEX REPLACE ".proto" ".pb.h"  HEADER_FILE ${PROTO_FILE})
    string(REGEX REPLACE ".proto" ".pb.cc" SOURCE_FILE ${PROTO_FILE})
    list(APPEND OUTPUT_FILES 
         ${CMAKE_CURRENT_BINARY_DIR}/${HEADER_FILE}
         ${CMAKE_CURRENT_BINARY_DIR}/${SOURCE_FILE})
    
    list(APPEND INPUT_FILES ${CMAKE_CURRENT_SOURCE_DIR}/${PROTO_FILE})
       
    message(STATUS "protoc:\n"
           "INPUT_FILES   : ${INPUT_FILES}\n"
           "OUTPUT_FILES  : ${OUTPUT_FILES}\n")
    
    set_source_files_properties(
        ${CMAKE_CURRENT_BINARY_DIR}/${HEADER_FILE} PROPERTIES GENERATED TRUE)        
    set_source_files_properties(
        ${CMAKE_CURRENT_BINARY_DIR}/${SOURCE_FILE} PROPERTIES GENERATED TRUE)
  endforeach()
    
  message(STATUS "custom command:\n"
          "OUTPUT  : ${OUTPUT_FILES}\n"
          "command : ${ProtobufCompiler_PATH} -I${CMAKE_CURRENT_SOURCE_DIR} "
                     "--cpp_out=${CMAKE_CURRENT_BINARY_DIR} ${INPUT_FILES}\n")
  add_custom_command(OUTPUT ${OUTPUT_FILES}
      COMMAND ${ProtobufCompiler_PATH} -I=${CMAKE_CURRENT_SOURCE_DIR} 
              --cpp_out=${CMAKE_CURRENT_BINARY_DIR} ${INPUT_FILES}
      DEPENDS ${PROTO_FILES})
endfunction()
