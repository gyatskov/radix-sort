#define _STR(x) #x
#define STR(x) _STR(x)
#define TODO(x) __pragma(message("TODO: "_STR(x) " :: " __FILE__ "@"STR(__LINE__)))

