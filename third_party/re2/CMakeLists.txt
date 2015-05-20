set(INSTALL_HFILES re2/filtered_re2.h
                   re2/re2.h
                   re2/set.h
                   re2/stringpiece.h
                   re2/variadic_function.h)

set(HFILES util/arena.h
           util/atomicops.h  
           util/benchmark.h  
           util/flags.h  
           util/logging.h  
           util/mutex.h  
           util/pcre.h  
           util/random.h  
           util/sparse_array.h  
           util/sparse_set.h  
           util/test.h  
           util/utf.h  
           util/util.h  
           util/valgrind.h  
           re2/filtered_re2.h  
           re2/prefilter.h  
           re2/prefilter_tree.h  
           re2/prog.h  
           re2/re2.h  
           re2/regexp.h  
           re2/set.h  
           re2/stringpiece.h  
           re2/testing/exhaustive_tester.h  
           re2/testing/regexp_generator.h  
           re2/testing/string_generator.h  
           re2/testing/tester.h  
           re2/unicode_casefold.h  
           re2/unicode_groups.h  
           re2/variadic_function.h  
           re2/walker-inl.h)

set(CCFILES util/arena.cc  
            util/hash.cc  
            util/rune.cc  
            util/stringpiece.cc  
            util/stringprintf.cc  
            util/strutil.cc  
            util/valgrind.cc  
            re2/bitstate.cc  
            re2/compile.cc  
            re2/dfa.cc  
            re2/filtered_re2.cc  
            re2/mimics_pcre.cc  
            re2/nfa.cc  
            re2/onepass.cc  
            re2/parse.cc  
            re2/perl_groups.cc  
            re2/prefilter.cc  
            re2/prefilter_tree.cc  
            re2/prog.cc  
            re2/re2.cc  
            re2/regexp.cc  
            re2/set.cc  
            re2/simplify.cc  
            re2/tostring.cc  
            re2/unicode_casefold.cc  
            re2/unicode_groups.cc)  

set(TESTCCFILES util/pcre.cc  
                util/random.cc  
                util/test.cc
                util/thread.cc  
                re2/testing/backtrack.cc  
                re2/testing/dump.cc  
                re2/testing/exhaustive_tester.cc  
                re2/testing/null_walker.cc  
                re2/testing/regexp_generator.cc  
                re2/testing/string_generator.cc  
                re2/testing/tester.cc)

set(TESTS charclass_test  
          compile_test  
          filtered_re2_test  
          mimics_pcre_test  
          parse_test  
          possible_match_test  
          re2_test  
          re2_arg_test  
          regexp_test  
          required_prefix_test  
          search_test  
          set_test  
          simplify_test  
          string_generator_test)

set(BIGTESTS dfa_test  
             exhaustive1_test  
             exhaustive2_test  
             exhaustive3_test  
             exhaustive_test  
             random_test)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

cc_library(re2
           HEADERS ${HFILES}
           SOURCES ${CCFILES})

cc_library(re2_testlib SOURCES ${TESTCCFILES}
           TARGET_DEPENDS re2)

set(LIBS
    ${CMAKE_THREAD_LIBS_INIT}
    ${THREADS_LIBRARIES})

foreach(test ${TESTS})
  cc_test(re2_${test}
           SOURCES re2/testing/${test}.cc
           TARGET_DEPENDS re2_testlib_static)
  target_link_libraries(re2_${test}_exe ${LIBS})
endforeach()

#foreach(TEST ${BIGTESTS})
#  add_executable(${TEST} re2/testing/${TEST}.cc)
#  target_link_libraries(${TEST} ${LIBS})
#endforeach()
