// The one translation unit that compiles dr_mp3 and dr_flac, from memory only. MP3 frames are decoded straight to
// float (no rounding to 16 bits inside the decoder); FLAC only in its own container (Ogg is refused before it).
#define DR_MP3_IMPLEMENTATION
#define DR_MP3_NO_STDIO
#define DR_MP3_FLOAT_OUTPUT
#include "dr_mp3.h"

#define DR_FLAC_IMPLEMENTATION
#define DR_FLAC_NO_STDIO
#define DR_FLAC_NO_OGG
#include "dr_flac.h"
