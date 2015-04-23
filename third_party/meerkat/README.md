# Meerkat Lightweight Embedded Web Server

Note(josh): this is commit:
    commit 5036446947c99f0ebd5785370888e28d88867998
    Author: Sergey Lyubka <valenok@gmail.com>
    Date:   Fri Apr 10 18:28:53 2015 +0100

    Apply mongoose patches


Lightweight Web Server for your application needs.
Turn anything into a web server fast.

- [Documentation](http://cesanta.com/docs.shtml) - configuration options and API reference
- [Examples](https://github.com/cesanta/mongoose/tree/master/examples) - example programs for various use cases

Check out Fossa - our [embedded multi-protocol library](https://github.com/cesanta/fossa) with TCP,UDP,HTTP,Websocket,MQTT,DNS support, designed for Internet Of Things!

# Features

- Works on Windows, Mac, UNIX/Linux, iPhone, Android eCos, QNX
and many other platforms
- Digest auth, URL rewrite, file blacklist
- Custom error pages, Virtual hosts, IP-based ACL, HTTP client
- Simple and clean
  [embedding API](meerkat.h). The source code is in single
  [meerkat.c](meerkat.c) file to make embedding easy
- Extremely lightweight, has a core of under 40kB and tiny runtime footprint
- Asynchronous, non-blocking core supporting single- or multi-threaded usage
- Stable, mature and tested, has several man-years invested
  in continuous improvement and refinement

Note: Meerkat does NOT support HTTPS, CGI, SSI, Websocket. So we advice to consider using [Mongoose Library](https://github.com/cesanta/mongoose) if you require this functionality.

# Contributions

People who have agreed to the
[Cesanta CLA](http://cesanta.com/contributors_la.html)
can make contributions. Note that the CLA isn't a copyright
_assigment_ but rather a copyright _license_.
You retain the copyright on your contributions.

# Licensing

Meerkat is released under commercial and
[GNU GPL v.2](http://www.gnu.org/licenses/old-licenses/gpl-2.0.html) open
source licenses. The GPLv2 open source License does not generally permit
incorporating this software into non-open source programs.
For those customers who do not wish to comply with the GPLv2 open
source license requirements,
[Cesanta](http://cesanta.com) offers a full,
royalty-free commercial license and professional support
without any of the GPL restrictions.
