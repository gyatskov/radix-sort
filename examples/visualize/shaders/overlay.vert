#version 450

layout(push_constant) uniform PC {
    float timeMs;
    float padX;  // right edge padding in NDC (e.g. 0.02)
    float padY;  // bottom edge padding in NDC
    float charW; // character width in NDC
    float charH; // character height in NDC
    uint  numChars;
} pc;

layout(location = 0) out vec2 fragUV;
layout(location = 1) flat out uint charIndex;

void main() {
    // 6 vertices per character quad (2 triangles), laid out left-to-right
    uint charIdx  = gl_VertexIndex / 6u;
    uint vertIdx  = gl_VertexIndex % 6u;

    // Triangle strip order: 0-1-2, 3-4-5
    vec2 offsets[6] = vec2[6](
        vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
        vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0)
    );
    vec2 uv = offsets[vertIdx];

    // Position: bottom-right corner, right-aligned
    float gap    = pc.charW * 0.3;  // inter-character spacing
    float stride = pc.charW + gap;
    float totalW = float(pc.numChars) * stride - gap;
    float x = (1.0 - pc.padX - totalW) + float(charIdx) * stride + uv.x * pc.charW;
    float y = (1.0 - pc.padY - pc.charH) + uv.y * pc.charH;

    gl_Position = vec4(x, y, 0.0, 1.0);
    fragUV      = uv;
    charIndex   = charIdx;
}
