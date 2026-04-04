#version 450

layout(push_constant) uniform PC {
    float value;
    float anchorX;  // NDC X of left edge of text block
    float anchorY;  // NDC Y of top edge of text block
    float charW;    // character width in NDC
    float charH;    // character height in NDC
    uint  numChars;
    uint  mode;     // 0 = time, 1 = integer
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

    float gap    = pc.charW * 0.3;  // inter-character spacing
    float stride = pc.charW + gap;
    float x = pc.anchorX + float(charIdx) * stride + uv.x * pc.charW;
    float y = pc.anchorY + uv.y * pc.charH;

    gl_Position = vec4(x, y, 0.0, 1.0);
    fragUV      = uv;
    charIndex   = charIdx;
}
