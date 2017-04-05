#version 430 core
in vec2 texFrag;
uniform sampler2D tex;
out vec4 color;
void main(){
    color = texture(tex,vec2(texFrag.x,1-texFrag.y));
}