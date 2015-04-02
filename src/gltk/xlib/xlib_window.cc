#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <GL/gl.h>
#include <GL/glx.h>

enum {
  X11_None = None
};

//Note: defined in X11/Xlib.h, but also a class in format.h
#undef None

#include <iostream>
#include "cppformat/format.h"

static const int kGlxContextMajorVersionARB = 0x2091;
static const int kGlxContextMinorversionARB = 0x2092;

typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig,
                                                     GLXContext, Bool,
                                                     const int*);

// Helper to check for extension string presence.  Adapted from:
//   http://www.opengl.org/resources/features/OGLextensions/
static bool IsExtensionSupported(const char *extList, const char *extension) {
  const char *start;
  const char *where, *terminator;

  /* Extension names should not have spaces. */
  where = strchr(extension, ' ');
  if (where || *extension == '\0')
    return false;

  /* It takes a bit of care to be fool-proof about parsing the
   OpenGL extensions string. Don't be fooled by sub-strings,
   etc. */
  for (start = extList;;) {
    where = strstr(start, extension);

    if (!where)
      break;

    terminator = where + strlen(extension);

    if (where == start || *(where - 1) == ' ')
      if (*terminator == ' ' || *terminator == '\0')
        return true;

    start = terminator;
  }

  return false;
}

static bool g_ctx_error_occurred = false;
static int HandleCtxErrorHandle(Display *dpy, XErrorEvent *ev) {
  g_ctx_error_occurred = true;
  return 0;
}

int main(int argc, char* argv[]) {
  Display *display = XOpenDisplay(NULL);

  if (!display) {
    fmt::print(std::cerr, "Failed to open X display\n");
    exit(1);
  }

  // Get a matching FB config
  static int visual_attribs[] = {
  GLX_X_RENDERABLE, True,
  GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
  GLX_RENDER_TYPE, GLX_RGBA_BIT,
  GLX_X_VISUAL_TYPE, GLX_TRUE_COLOR,
  GLX_RED_SIZE, 8,
  GLX_GREEN_SIZE, 8,
  GLX_BLUE_SIZE, 8,
  GLX_ALPHA_SIZE, 8,
  GLX_DEPTH_SIZE, 24,
  GLX_STENCIL_SIZE, 8,
  GLX_DOUBLEBUFFER, True,
  //GLX_SAMPLE_BUFFERS  , 1,
  //GLX_SAMPLES         , 4,
      X11_None };

  int glx_major, glx_minor;

  // FBConfigs were added in GLX version 1.3.
  if (!glXQueryVersion(display, &glx_major, &glx_minor)
      || ((glx_major == 1) && (glx_minor < 3)) || (glx_major < 1)) {
    fmt::print(std::cerr, "Invalid GLX version");
    exit(1);
  }

  printf( "Getting matching framebuffer configs\n" );
  int fbcount;
  GLXFBConfig* fbc = glXChooseFBConfig(display, DefaultScreen(display),
                                       visual_attribs, &fbcount);
  if (!fbc) {
    fmt::print(std::cerr, "Failed to retrieve a framebuffer config\n");
    exit(1);
  }
  fmt::print("Found {} matching FB configs.\n", fbcount);

  // Pick the FB config/visual with the most samples per pixel
  fmt::print("Getting XVisualInfos\n");
  int best_fbc = -1;
  int worst_fbc = -1;
  int best_num_samp = -1;
  int worst_num_samp = 999;

  for (int i = 0; i < fbcount; ++i) {
    XVisualInfo *vi = glXGetVisualFromFBConfig(display, fbc[i]);
    if (vi) {
      int samp_buf, samples;
      glXGetFBConfigAttrib(display, fbc[i], GLX_SAMPLE_BUFFERS, &samp_buf);
      glXGetFBConfigAttrib(display, fbc[i], GLX_SAMPLES, &samples);

      fmt::print("  Matching fbconfig {}, visual ID {#2x}: SAMPLE_BUFFERS = {},"
                 " SAMPLES = {}\n", i, vi->visualid, samp_buf, samples);

      if (best_fbc < 0 || samp_buf && samples > best_num_samp) {
        best_fbc = i;
        best_num_samp = samples;
      }
      if (worst_fbc < 0 || !samp_buf || samples < worst_num_samp) {
        worst_fbc = i;
        worst_num_samp = samples;
      }
    }
    XFree(vi);
  }

  GLXFBConfig bestFbc = fbc[best_fbc];

  // Be sure to free the FBConfig list allocated by glXChooseFBConfig()
  XFree(fbc);

  // Get a visual
  XVisualInfo *vi = glXGetVisualFromFBConfig(display, bestFbc);
  fmt::print("Chosen visual ID = {#x}\n", vi->visualid );

  fmt::print("Creating colormap\n");
  XSetWindowAttributes swa;
  Colormap cmap;
  swa.colormap = cmap = XCreateColormap(display,
                                        RootWindow(display, vi->screen),
                                        vi->visual, AllocNone);
  swa.background_pixmap = X11_None;
  swa.border_pixel = 0;
  swa.event_mask = StructureNotifyMask;

  fmt::print("Creating window\n");
  Window win = XCreateWindow(display, RootWindow(display, vi->screen), 0, 0,
                             100, 100, 0, vi->depth, InputOutput, vi->visual,
                             CWBorderPixel | CWColormap | CWEventMask,
                             &swa);
  if (!win) {
    fmt::print(std::cerr, "Failed to create window.\n");
    exit(1);
  }

  // Done with the visual info data
  XFree( vi );

  XStoreName(display, win, "GL 3.0 Window");

  fmt::print("Mapping window\n");
  XMapWindow(display, win);

  // Get the default screen's GLX extension list
  const char *glx_extentions = glXQueryExtensionsString(display,
                                                        DefaultScreen(display));

  // NOTE: It is not necessary to create or make current to a context before
  // calling glXGetProcAddressARB
  glXCreateContextAttribsARBProc glX_create_context_attribs_ARB = 0;
  glX_create_context_attribs_ARB =
      (glXCreateContextAttribsARBProc) glXGetProcAddressARB(
          (const GLubyte *) "glX_create_context_attribs_ARB");

  GLXContext ctx = 0;

  // Install an X error handler so the application won't exit if GL 3.0
  // context allocation fails.
  //
  // Note this error handler is global.  All display connections in all threads
  // of a process use the same error handler, so be sure to guard against other
  // threads issuing X commands while this code is running.
  g_ctx_error_occurred = false;
  int (*old_handler)(Display*,
                     XErrorEvent*) = XSetErrorHandler(&HandleCtxErrorHandle);

  // Check for the GLX_ARB_create_context extension string and the function.
  // If either is not present, use GLX 1.3 context creation method.
  if (!IsExtensionSupported(glx_extentions, "GLX_ARB_create_context")
      || !glX_create_context_attribs_ARB) {
    fmt::print(std::cerr, "glXCreateContextAttribsARB() not found"
               " ... using old-style GLX context\n");
    ctx = glXCreateNewContext(display, bestFbc, GLX_RGBA_TYPE, 0, True);
  }

  // If it does, try to get a GL 3.0 context!
  else {
    int context_attribs[] = {
        kGlxContextMajorVersionARB, 3,
        kGlxContextMinorversionARB, 0,
        //GLX_CONTEXT_FLAGS_ARB        , GLX_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB,
        X11_None };

    fmt::print("Creating context\n");
    ctx = glX_create_context_attribs_ARB(display, bestFbc, 0, True,
                                         context_attribs);

    // Sync to ensure any errors generated are processed.
    XSync(display, False);
    if (!g_ctx_error_occurred && ctx)
      fmt::print("Created GL 3.0 context\n");
    else {
      // Couldn't create GL 3.0 context.  Fall back to old-style 2.x context.
      // When a context version below 3.0 is requested, implementations will
      // return the newest context version compatible with OpenGL versions less
      // than version 3.0.
      // kGlxContextMajorVersionARB = 1
      context_attribs[1] = 1;
      // kGlxContextMinorversionARB = 0
      context_attribs[3] = 0;

      g_ctx_error_occurred = false;

      fmt::print("Failed to create GL 3.0 context"
                 " ... using old-style GLX context\n");
      ctx = glX_create_context_attribs_ARB(display, bestFbc, 0, True,
                                           context_attribs);
    }
  }

  // Sync to ensure any errors generated are processed.
  XSync(display, False);

  // Restore the original error handler
  XSetErrorHandler(old_handler);

  if (g_ctx_error_occurred || !ctx) {
    fmt::print("Failed to create an OpenGL context\n");
    exit(1);
  }

  // Verifying that context is a direct context
  if (!glXIsDirect(display, ctx)) {
    fmt::print("Indirect GLX rendering context obtained\n");
  } else {
    fmt::print("Direct GLX rendering context obtained\n");
  }

  fmt::print("Making context current\n");
  glXMakeCurrent( display, win, ctx );

  glClearColor( 0, 0.5, 1, 1 );
  glClear( GL_COLOR_BUFFER_BIT );
  glXSwapBuffers ( display, win );

  sleep( 1 );

  glClearColor ( 1, 0.5, 0, 1 );
  glClear ( GL_COLOR_BUFFER_BIT );
  glXSwapBuffers ( display, win );

  sleep( 1 );

  glXMakeCurrent( display, 0, 0 );
  glXDestroyContext( display, ctx );

  XDestroyWindow( display, win );
  XFreeColormap( display, cmap );
  XCloseDisplay( display );

  return 0;
}
