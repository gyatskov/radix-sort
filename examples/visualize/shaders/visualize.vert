#version 450

layout(set = 0, binding = 0) readonly buffer DataBuffer {
    uint values[];
} data;

layout(push_constant) uniform PushConstants {
    uint  count;
    float maxValue;
    float yOffset;   // vertical center of this row
    float yScale;    // vertical extent
    float xOffset;   // horizontal center of this column
    float xScale;    // horizontal extent (0..1, fraction of full width)
} pc;

layout(location = 0) out float outNormValue;

void main() {
    uint  idx = gl_VertexIndex;
    float nv  = float(data.values[idx]) / pc.maxValue;

    // X: spread across the column with small margins
    float t = float(idx) / float(pc.count - 1u);
    float x = pc.xOffset + pc.xScale * mix(-0.95, 0.95, t);

    // Y: map normalised value into the assigned half of the screen
    //    In Vulkan clip space Y points downward, so "up" is negative.
    //    nv == 0  ->  bottom of the section  (yOffset + yScale)
    //    nv == 1  ->  top of the section     (yOffset - yScale)
    float y = pc.yOffset + pc.yScale * (1.0 - 2.0 * nv);

    gl_Position = vec4(x, y, 0.0, 1.0);
    gl_PointSize = 1.0;
    outNormValue = nv;
}
