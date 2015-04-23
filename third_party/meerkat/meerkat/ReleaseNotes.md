# Mongoose Release Notes

## Release 5.6, 2015-03-17

- Added `-dav_root` configuration option that gives an ability to mount
  a different root directory (not document_root)
- Fixes for build under Win23 and MinGW
- Bugfix: Double dots removal
- Bugfix: final chunked response double-send
- Fixed compilation in 64-bit environments
- Added OS/2 compatibility
- Added `getaddrinfo()` call and `NS_ENABLE_GETADDRINFO`
- Added integer overflow protection in `iobuf_append()` and `deliver_websocket_frame()`
- Fixed NetBSD build
- Enabled `NS_ENABLE_IPV6` build for Visual Studio 2008+
- Enhanced comma detection in `parse_header()`
- Fixed unchanged memory accesses on ARM
- Added ability to use custom memory allocator through NS_MALLOC, NS_FREE, NS_REALLOC

## Release 5.5, 2014-10-28

Initial release
