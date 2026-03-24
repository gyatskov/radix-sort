#version 450

layout(push_constant) uniform PC {
    float timeMs;
    float padX;
    float padY;
    float charW;
    float charH;
    uint  numChars;
} pc;

layout(location = 0) in vec2 fragUV;
layout(location = 1) flat in uint charIndex;

layout(location = 0) out vec4 outColor;

// 5x5 bitmap font for digits 0-9, '.', 'm', 's', space
// Each glyph is 5 columns x 5 rows, packed into 25 bits (LSB = top-left).
const uint GLYPHS[14] = uint[14](
    0xE8C62Eu,  // 0: 01110  10001  10001  10001  01110
    0xE21184u,  // 1: 00100  01100  00100  00100  01110
    0x1F41A2Eu, // 2: 01110  10001  00110  01000  11111
    0xE0B82Eu,  // 3: 01110  00001  01110  00001  01110
    0x10FE31u,  // 4: 10001  10001  11111  00001  00001
    0x1E0FA1Fu, // 5: 11111  10000  11110  00001  11110
    0xE8FA0Eu,  // 6: 01110  10000  11110  10001  01110
    0x42083Fu,  // 7: 11111  00001  00010  00100  00100
    0xE8BA2Eu,  // 8: 01110  10001  01110  10001  01110
    0xE0BE2Eu,  // 9: 01110  10001  01111  00001  01110
    0x400000u,  // .: 00000  00000  00000  00000  00100
    0x118D771u, // m: 10001  11011  10101  10001  10001
    0x1E0BA0Fu, // s: 01111  10000  01110  00001  11110
    0x00000u    // space
);

// Decode the time value into individual character indices.
// Format: "NNN.NN ms" (9 characters, index 0..8)
uint getCharGlyphIndex(uint pos) {
    // Decompose timeMs into integer + fractional
    float t = clamp(pc.timeMs, 0.0, 999.99);
    uint intPart  = uint(t);
    uint fracPart = uint(fract(t) * 100.0 + 0.5);

    // Digit layout: [H][T][U][.][d1][d2][ ][m][s]
    //                0  1  2  3   4   5  6  7  8
    if (pos == 0u) return intPart / 100u;
    if (pos == 1u) return (intPart / 10u) % 10u;
    if (pos == 2u) return intPart % 10u;
    if (pos == 3u) return 10u; // dot
    if (pos == 4u) return fracPart / 10u;
    if (pos == 5u) return fracPart % 10u;
    if (pos == 6u) return 13u; // space
    if (pos == 7u) return 11u; // m
    if (pos == 8u) return 12u; // s
    return 13u;
}

void main() {
    uint glyphIdx = getCharGlyphIndex(charIndex);
    uint glyph    = GLYPHS[glyphIdx];

    // Map UV to 5x5 grid cell
    float cellX = fragUV.x * 5.0;
    float cellY = fragUV.y * 5.0;
    int col = clamp(int(cellX), 0, 4);
    int row = clamp(int(cellY), 0, 4);

    // Intra-cell margins: shrink lit area to leave gaps between pixels
    float fracX = fract(cellX);
    float fracY = fract(cellY);
    float margin = 0.15;
    bool inPixel = fracX > margin && fracX < (1.0 - margin)
                && fracY > margin && fracY < (1.0 - margin);

    // Look up bit: row-major, MSB = top-left (column is mirrored)
    uint bitIdx = uint(row) * 5u + (4u - uint(col));
    bool lit = inPixel && (glyph & (1u << bitIdx)) != 0u;

    if (lit)
        outColor = vec4(1.0, 1.0, 1.0, 1.0);
    else
        outColor = vec4(0.0, 0.0, 0.0, 0.5); // semi-transparent background
}
