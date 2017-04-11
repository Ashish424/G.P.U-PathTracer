#version 430 core
in vec2 texFrag;
uniform usampler2D tex;
out vec4 color;
void main(){
    color = texture(tex,vec2(texFrag.x,1-texFrag.y));
    color/=255.0;
}