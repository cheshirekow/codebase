# TODO: see Makefile.am in this directory, observe how the lcm build system
# unpacks dependency jars and then repacks lcm.jar. We should do the same

set(jchart2d_src_dir ${CMAKE_CURRENT_SOURCE_DIR}/jchart2d-code/src)
set(jchart2d_ext_dir = ${CMAKE_CURRENT_SOURCE_DIR}/jchart2d-code/ext)
set(jide_jar = ${jchart2d_ext_dir}/jide-oss-2.9.7.jar)
set(xmlgraphics_jar = ${jchart2d_ext_dir}/xmlgraphics-commons-1.3.1.jar)

set(lcm_java_files
    lcm/lcm/LCMEncodable.java
    lcm/lcm/LCM.java
    lcm/lcm/LCMSubscriber.java
    lcm/lcm/URLParser.java
    lcm/lcm/MemqProvider.java
    lcm/lcm/Provider.java
    lcm/lcm/TCPProvider.java
    lcm/lcm/UDPMulticastProvider.java
    lcm/lcm/TCPService.java
    lcm/lcm/LogFileProvider.java
		lcm/lcm/TCPProvider.java
		lcm/lcm/MessageAggregator.java
		lcm/lcm/LCMDataInputStream.java
		lcm/lcm/LCMDataOutputStream.java
    lcm/util/BufferedRandomAccessFile.java
    lcm/util/ColorMapper.java
    lcm/util/TableSorter.java
    lcm/util/ParameterListener.java
    lcm/util/ParameterGUI.java
    lcm/util/JImage.java
    lcm/util/ClassDiscoverer.java
    lcm/logging/JScrubber.java
    lcm/logging/JScrubberListener.java
    lcm/logging/Log.java
    lcm/logging/LogDiagnostic.java
    lcm/logging/LogPlayer.java
    lcm/spy/Spy.java
    lcm/spy/SpyPlugin.java
    lcm/spy/LCMTypeDatabase.java
    lcm/spy/ObjectPanel.java
    lcm/spy/ChartData.java
    lcm/spy/ChannelData.java
    lcm/spy/ZoomableChartScrollWheel.java
    lcm/spy/ChannelData.java)

set(CMAKE_JAVA_COMPILE_FLAGS -source 1.5 -target 1.5)
set(CMAKE_JAVA_INCLUDE_PATH ${jchart2d_src_dir})

add_jar(lcm_jar SOURCES ${lcm_java_files}
                INCLUDE_JARS $(jide_jar) $(xmlgraphics_jar)
                #[ENTRY_POINT entry]
                VERSION 1.2.2
                OUTPUT_NAME lcm)

