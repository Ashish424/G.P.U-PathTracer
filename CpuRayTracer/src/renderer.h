#ifndef RENDERER_H
#define RENDERER_H


#include "scene.h"
#include "camera.h"


#include <glad/glad.h>

class GLFWwindow;
class Renderer {

public:
    Renderer(Scene *scene, Camera *camera);

    void render(int samples=4);
    void save_image(const char * file_path);
    void runInLoop();
    ~Renderer();


private:
    Scene *scene;
    Camera *camera;
    glm::u8vec3 *pixelBuffer;
    GLFWwindow * mainWindow = nullptr;
    void update(double delta);
    //render to image
    //draw to quad to display by opengl
    void draw();
    int width,height;
    GLuint textureImage;
    GLuint quadVao = 0,quadVbo;
    GLuint quadProgram = 0;


};

#endif //RENDERER_H