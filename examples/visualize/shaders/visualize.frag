#version 450

layout(location = 0) in float inNormValue;
layout(location = 0) out vec4  outColor;

void main() {
    // Heat-map: blue -> cyan -> green -> yellow -> red
    float t = clamp(inNormValue, 0.0, 1.0);
    vec3 color;
    if (t < 0.25)
        color = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), t * 4.0);
    else if (t < 0.5)
        color = mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), (t - 0.25) * 4.0);
    else if (t < 0.75)
        color = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), (t - 0.5) * 4.0);
    else
        color = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), (t - 0.75) * 4.0);

    outColor = vec4(color, 1.0);
}
