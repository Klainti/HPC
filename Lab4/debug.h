#ifndef DEBUG_H_INCLUDED
#define DEBUG_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>

#ifdef DEBUG_COLOR /* Colorize with ANSI escape sequences. */

#  define DEBUG_COLOR_RED     "\x1b[31m"
#  define DEBUG_COLOR_YELLOW  "\x1b[33m"
#  define DEBUG_COLOR_BLUE    "\x1b[34m"
#  define DEBUG_COLOR_RESET   "\x1b[0m"

#else

#  define DEBUG_COLOR_RED     ""
#  define DEBUG_COLOR_YELLOW  ""
#  define DEBUG_COLOR_BLUE    ""
#  define DEBUG_COLOR_RESET ""

#endif /*colors*/

#define debug_prefix(tag,color) printf(color "Debug[%s|%d|%s]: " tag DEBUG_COLOR_RESET, __FILE__, __LINE__, __func__)

#ifdef DEBUG

#  define debug(...)      do { debug_prefix("",          DEBUG_COLOR_BLUE);   printf(__VA_ARGS__); printf("\n"); } while(0)
#  define debug_w(...)    do { debug_prefix("WARNING: ", DEBUG_COLOR_YELLOW); printf(__VA_ARGS__); printf("\n"); } while(0)
#  define debug_e(...)    do { debug_prefix("ERROR: ",   DEBUG_COLOR_RED);    printf(__VA_ARGS__); printf("\n"); } while(0)

#else

#  define debug(...)      ((void) 0)
#  define debug_w(...)    ((void) 0)
#  define debug_e(...)    ((void) 0)

#endif /* DEBUG */
#endif /* DEBUG_H_INCLUDED */
