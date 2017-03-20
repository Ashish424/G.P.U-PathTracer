#include <vector>
#include <stdio.h>
#include <iostream>

#include "renderer.h"
#include "../lib/lodepng/lodepng.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>

// Clamp double to min/max of 0/1
inline double clamp(double x){ return x<0 ? 0 : x>1 ? 1 : x; }
// Clamp to between 0-255
inline int toInt(double x){ return int(clamp(x)*255+.5); }
const GLchar* vertexShaderQuad = "#version 430 core\n"
        "layout (location = 0) in vec3 position;\n"
        "layout (location = 1) in vec2 texCoord;\n"
        "out vec2 texFrag;\n"

        "void main()\n"
        "{\n"
        "gl_Position = vec4(position, 1.0f);\n"
        "texFrag = texCoord;\n"
        "}\n\0";


const GLchar* fragmentShaderQuad = "#version 430 core\n"
        "in vec2 texFrag;\n"
        "uniform sampler2D tex;\n"
        "out vec4 color;\n"
        "void main()\n"
        "{\n"
        "color = texture(tex,vec2(texFrag.x,1-texFrag.y));\n"
        "}\n\0";


GLuint makeProgram(GLuint vS, GLuint fS,bool deleteDetachShaders);
GLuint compileShader(GLenum shaderType,const char * source);

Renderer::Renderer(Scene *scene, Camera *camera) {
    this->scene = scene;
    this->camera = camera;
    pixelBuffer = new glm::u8vec3[camera->get_width()*camera->get_height()];
    width = camera->get_width();
    height = camera->get_height();


    {
        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_SAMPLES, 0);
        glfwWindowHint(GLFW_RED_BITS, 8);
        glfwWindowHint(GLFW_GREEN_BITS, 8);
        glfwWindowHint(GLFW_BLUE_BITS, 8);
        glfwWindowHint(GLFW_ALPHA_BITS, 8);
        glfwWindowHint(GLFW_STENCIL_BITS, 8);
        glfwWindowHint(GLFW_DEPTH_BITS, 24);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

#ifdef MY_DEBUG_OPENGL_APP
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT,GLFW_TRUE);
#endif


        using namespace std;

// Create a GLFWwindow object
        mainWindow = glfwCreateWindow(camera->get_width(), camera->get_height(), "Cpu Path Tracer", nullptr,
                                      nullptr);
        if (mainWindow == nullptr) {
            std::cout << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            exit(-1);
        }
        glfwMakeContextCurrent(mainWindow);
        glfwSetWindowUserPointer(mainWindow,this);

        glfwSetKeyCallback(mainWindow,[](GLFWwindow *window, int key, int scancode, int action, int mods){
           if (key == GLFW_KEY_ESCAPE)
               glfwSetWindowShouldClose(window, GL_TRUE);

        });


        if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
            cout << "Failed to initialize OpenGL context" << endl;
            exit(-1);
        }

    }

    //setup quad
    {
        GLfloat quadVertices[20] = {
                // Positions        // Texture Coords
                -1.0f,  1.0f, 0.99f, 0.0f, 1.0f,
                -1.0f, -1.0f, 0.99f, 0.0f, 0.0f,
                1.0f,  1.0f,  0.99f, 1.0f, 1.0f,
                1.0f, -1.0f,  0.99f, 1.0f, 0.0f,
        };

        // Setup plane VAO
        glGenVertexArrays(1, &quadVao);
        glGenBuffers(1, &quadVbo);

        glBindVertexArray(quadVao);
        glBindBuffer(GL_ARRAY_BUFFER, quadVbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
        glBindVertexArray(0);


    }
    //setup texture
    {
        glGenTextures(1, &textureImage);
        glBindTexture(GL_TEXTURE_2D, textureImage);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE,pixelBuffer);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);


    }
    //compile and link shaders
    {
        auto vertS = compileShader(GL_VERTEX_SHADER,vertexShaderQuad);
        auto fragS = compileShader(GL_FRAGMENT_SHADER,fragmentShaderQuad);
        quadProgram = makeProgram(vertS,fragS,true);
    }
}

void Renderer::runInLoop() {




    double last = 0;
    glfwSetTime(last);
    double delta = 0.0f;
    while (!glfwWindowShouldClose(mainWindow)) {

        double curr = glfwGetTime();
        delta = curr-last;
        last = curr;

        glfwPollEvents();


        using namespace std;
        cout <<delta << endl;
        update(delta);

        //TODO add sample logic here
        render(10);
        draw();
        glfwSwapBuffers(mainWindow);

    }





}
void Renderer::update(double delta){



}

void Renderer::draw() {
    //TODO use a faster implementation here

    glClearColor(1.0, 1.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureImage);

//  Update texture on GPU
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width,height, GL_RGB, GL_UNSIGNED_BYTE,pixelBuffer);


    glUseProgram(quadProgram);
    glUniform1i(glGetUniformLocation(quadProgram, "tex"), 0);

    glBindVertexArray(quadVao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

}

void Renderer::render(int samples) {
    int width = camera->get_width();
    int height = camera->get_height();
    double samples_recp = 1./samples;
    std::cout << samples << std::endl;

    // Main Loop
    #pragma omp parallel for schedule(dynamic, 1)       // OpenMP
    for (int y=0; y<height; y++){
        unsigned short Xi[3]={0,0,y*y*y};               // Stores seed for erand48

//        fprintf(stderr, "\rRendering (%i samples): %.2f%% ",      // Prints
//                samples, (double)y/height*100);                   // progress

        for (int x=0; x<width; x++){
            glm::dvec3 col = glm::dvec3();

            for (int a=0; a<samples; a++){
                Ray ray = camera->get_ray(x, y, a>0, Xi);
                col = col + scene->trace_ray(ray,0,Xi);
            }

            pixelBuffer[(y)*width + x].x = toInt((col * samples_recp).x);
            pixelBuffer[(y)*width + x].y = toInt((col * samples_recp).y);
            pixelBuffer[(y)*width + x].z = toInt((col * samples_recp).z);

        }
    }
}

void Renderer::save_image(const char *file_path) {
    int width = camera->get_width();
    int height = camera->get_height();

    std::vector<unsigned char> pixel_buffer;

    int pixel_count = width*height;

    for (int i=0; i<pixel_count; i++) {
        pixel_buffer.push_back(pixelBuffer[i].x);
        pixel_buffer.push_back(pixelBuffer[i].y);
        pixel_buffer.push_back(pixelBuffer[i].z);
        pixel_buffer.push_back(255);
    }

    //Encode the image
    unsigned error = lodepng::encode(file_path, pixel_buffer, width, height);
    //if there's an error, display it
    if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;

    pixel_buffer.clear();
}


//TODO move to common code for cpu and gpu
GLuint compileShader(GLenum shaderType,const char * source){

    //TODO see everything ok here
    GLuint Shader = glCreateShader(shaderType);
    glShaderSource(Shader, 1, &source, NULL);
    glCompileShader(Shader);
// Check for compile time errors
    GLint success;
    GLchar infoLog[512];
    glGetShaderiv(Shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(Shader, 512, NULL, infoLog);
        //TODO pass info to this function to know what type of error here
        std::cerr << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    return Shader;


}

GLuint makeProgram(GLuint vS, GLuint fS,bool deleteDetachShaders){
    GLint success = 0;
    GLchar infoLog[512];
// Link shaders
    GLuint sp = glCreateProgram();
    glAttachShader(sp, vS);
    glAttachShader(sp, fS);
    glLinkProgram(sp);
// Check for linking errors
    glGetProgramiv(sp, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(sp, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    if(deleteDetachShaders) {
        glDetachShader(sp,vS);glDeleteShader(vS);
        glDetachShader(sp,fS);glDeleteShader(fS);
    }
    return sp;
}

Renderer::~Renderer() {
    glDeleteVertexArrays(1, &quadVao);
    glDeleteBuffers(1, &quadVbo);

    glDeleteProgram(quadProgram);

}
