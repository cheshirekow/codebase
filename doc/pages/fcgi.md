C++ Bindings for fact-cgi {#fcgi_page}
==========

cpp-fcgi provides a thin object oriented wrapper around the thread-safe
fast CGI library API. The library is rather object oriented to begin with,
so for the most part it just provides member methods which wrap the
object methods from the C library. There isn't any added functionality,
it just makes code slightly more readable.

Below is a simple example of it's usage:

@code{cc}

#include <cpp-fcgi.h>
#include <iostream>

int main( int argc, char** argv )
{
    if( argc < 2 )
    {
        std::cerr << "Usage: " << argv[0] << " [SOCKETPATH]|:[PORTNUMBER] \n";
        return 0;
    }

    fcgi::Socket  socket;
    fcgi::Request request;

    // initialize the library (same as FCGX_Init() )
    int result = fcgi::init();
    if( result != 0 )
    {
        std::cerr << "Failed to initialize FCGI" << std::endl;
        return 0;
    }

    // open the socket (same as FCGX_OpenSocket )
    result = socket.open(argv[1],100);
    if( result < 0 )
    {
        std::cerr << "Failed to open socket: " << argv[1] << ", error: "
                  << result << "\n";
        return 0;
    }

    // initialize the request object and associate it with the opened socket
    if( request.init(socket) != 0 )
    {
        socket.close();
        std::cerr << "Failed to initialize the request\n";
        return 0;
    }

    // just a count of requests processed
    int reqNum = 0;

    // loop and receive requests
    while( true )
    {
        // accept a request
        result = request.accept();

        // check the result
        if( result != 0 )
        {
            std::cerr << "Failed to accept request: " << result << std::endl;
            break;
        }

        // write a simple message
        request.out() << "Content-type: text/html\n"
            << "\n"
            << "<title>FastCGI Hello! (C, fcgiapp library)</title>"
            << "<h1>FastCGI Hello! (C, fcgiapp library)</h1>"
            << "Request number " << ++reqNum
            << " running on host <i>"
            << request.getParam("SERVER_NAME")
            << "</i> Process ID: "
            << getpid()
            << "\n";

        // free request memory and prepare for the next one, note this is
        // not stricly necessary as it will happen automatically on the next
        // accept() call if the request hasn't been finished yet
        request.finish();
    }

    request.free(); //< free hidden memory of the request object
    socket.close(); //< close the socket
}

@endcode