#version 450

layout(set = 0, binding = 0) uniform sampler s_Color;
layout(set = 0, binding = 1) uniform texture2D t_Color;

layout(location = 0) in vec2 v_Uv;
layout(location = 1) in vec4 v_Color;
layout(location = 2) flat in uint v_Mode;

layout(location = 0) out vec4 Target0;

const uint MODE_TEXT = 0;
const uint MODE_IMAGE = 1;
const uint MODE_GEOMETRY = 2;

void main() {
    switch(v_Mode) {
        case MODE_TEXT: {
            Target0 = v_Color * vec4(1.0, 1.0, 1.0, texture(sampler2D(t_Color, s_Color), v_Uv).r);
            break;
        }
        case MODE_IMAGE: {
            Target0 = texture(sampler2D(t_Color, s_Color), v_Uv);
            break;
        }
        case MODE_GEOMETRY: {
            Target0 = v_Color;
            break;
        }
    }
}