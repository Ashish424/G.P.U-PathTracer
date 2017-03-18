//
// Created by ashish on 3/7/17.
//

#include <glad/glad.h>
#include "Scene.hpp"
#include <QKeyEvent>
#include <QDebug>
#include <QOpenGLContext>



const GLchar* vertexShaderSource = "#version 430 core\n"
        "layout (location = 0) in vec3 position;\n"
        "void main()\n"
        "{\n"
        "gl_Position = vec4(position.x, position.y, position.z, 1.0);\n"
        "}\0";
const GLchar* fragmentShaderSource = "#version 430 core\n"
        "out vec4 color;\n"
        "void main()\n"
        "{\n"
        "color = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
        "}\n\0";

GLuint VBO, VAO,VBO2,VAO2;
GLuint shaderProgram;

void APIENTRY openglCallbackFunction(GLenum source,
                                     GLenum type,
                                     GLuint id,
                                     GLenum severity,
                                     GLsizei length,
                                     const GLchar* message,
                                     const void* userParam);

void initShaderAndBuffers();
Scene::Scene(QWidget *parent):QOpenGLWidget(parent)
{
//    QObject::connect(timer,SIGNAL(timeout()),this,SLOT(FixedUpdate()));
//    timer->start(timestep);
//    setFocusPolicy(Qt::FocusPolicy::ClickFocus);
}



void Scene::initializeGL()
{



    if (!gladLoadGL())
    {
        qDebug() << "Failed to initialize context" ;

    }



    QOpenGLContext *cont = context();
    qDebug() << "Context valid: " << cont->isValid();
    qDebug() << "Really used OpenGl: " << cont->format().majorVersion() << "." << cont->format().minorVersion();
    qDebug() << "OpenGl information: VENDOR:       " << (const char*)glGetString(GL_VENDOR);
    qDebug() << "                    RENDERDER:    " << (const char*)glGetString(GL_RENDERER);
    qDebug() << "                    VERSION:      " << (const char*)glGetString(GL_VERSION);
    qDebug() << "                    GLSL VERSION: " << (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION);
    qDebug() << "endstuff\n";




    glClearColor(0.0f,0.0f,0.0f,1.0f);

    if(glDebugMessageCallback){
        qDebug() << "Registered OpenGL debug callback ";
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        glDebugMessageCallback(openglCallbackFunction, nullptr);
        GLuint unusedIds = 0;
        glDebugMessageControl(GL_DONT_CARE,
                              GL_DONT_CARE,
                              GL_DONT_CARE,
                              0,
                              &unusedIds,
                              GL_TRUE);
    }
    else
        qDebug() << "glDebugMessageCallback not available";






    initShaderAndBuffers();


}


void Scene::resizeGL(int w, int h)
{

    glViewport(0,0,w,h);
    //TODO update matrix here
}

void Scene::paintGL()
{

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

//    glEnable(GL_DEPTH_TEST);

// Accept fragment if it closer to the camera than the former one
    //TODO see less equal vs less
//    glDepthFunc(GL_LEQUAL);


//    glViewport(0,0,fWidth,fHeight);

    glClearColor(211/256.0f, 211/256.0f,211/256.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT);





    //draw your stuff
    glUseProgram(shaderProgram);
    glBindVertexArray(VAO);
    glDrawArrays(GL_LINE_LOOP,0, 4);
    glBindVertexArray(0);

//    glUseProgram(shaderProgram);
//    glBindVertexArray(VAO2);
//    glDrawArrays(GL_LINE_STRIP,0, 100);
    glBindVertexArray(0);





}

Scene::~Scene() {


    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);

    glDeleteVertexArrays(1, &VAO2);
    glDeleteBuffers(1, &VBO2);

}

//void Scene::FixedUpdate() {


    //cv::imshow("PerlinNoise",img);
//    slideAhead+=0.00016;

//    this->repaint();



//}

void initShaderAndBuffers() {



    GLfloat vertices[] = {
            -0.5f,0.5f,0.0f,
            0.5f,0.5f,0.0f,
            0.5f,-0.5f,0.0f,
            -0.5f,-0.5f,0.0f
    };

//    for(size_t i =0;i<sizeof(vertices)/sizeof(GLfloat);++i)
//        vertices[i]*=2;


    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
// Check for compile time errors
    GLint success;
    GLchar infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        qDebug() << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog;
    }
// Fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
// Check for compile time errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        qDebug() << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog;
    }
// Link shaders
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
// Check for linking errors
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        qDebug() << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);


    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);


    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER,0);





    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER,VBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid *) 0);
    glEnableVertexAttribArray(0);


    glBindVertexArray(0);







    std::vector<GLfloat> cVerts(3*100,0.0f);
    GLfloat controlPoints[] = {-1,-1,0.5,0,-1,1};
    for (int i = 0; i < cVerts.size(); i+=3) {
        float p =  i*1.0f/cVerts.size();
        cVerts[i]   = (1-p)*(1-p)*controlPoints[0]+2*(p)*(1-p)*controlPoints[2]+(p)*(p)*controlPoints[4];
        cVerts[i+1] = (1-p)*(1-p)*controlPoints[1]+2*(p)*(1-p)*controlPoints[3]+(p)*(p)*controlPoints[5];


    }



    glGenVertexArrays(1,&VAO2);
    glGenBuffers(1,&VBO2);

    glBindBuffer(GL_ARRAY_BUFFER,VBO2);
    glBufferData(GL_ARRAY_BUFFER,sizeof(GLfloat)*cVerts.size(),&cVerts[0],GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER,0);

    glBindVertexArray(VAO2);
    glBindBuffer(GL_ARRAY_BUFFER,VBO2);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,3*sizeof(GLfloat), nullptr);
    glBindVertexArray(0);











}


void APIENTRY openglCallbackFunction(GLenum source,
                                     GLenum type,
                                     GLuint id,
                                     GLenum severity,
                                     GLsizei length,
                                     const GLchar* message,
                                     const void* userParam){

    using namespace std;
    qDebug() << "---------------------opengl-callback-start------------" ;
    qDebug() << "message: "<< message ;
    qDebug() << "type: ";
    switch (type) {
        case GL_DEBUG_TYPE_ERROR:
            qDebug() << "ERROR";
            break;
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
            qDebug() << "DEPRECATED_BEHAVIOR";
            break;
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
            qDebug() << "UNDEFINED_BEHAVIOR";
            break;
        case GL_DEBUG_TYPE_PORTABILITY:
            qDebug() << "PORTABILITY";
            break;
        case GL_DEBUG_TYPE_PERFORMANCE:
            qDebug() << "PERFORMANCE";
            break;
        case GL_DEBUG_TYPE_OTHER:
            qDebug() << "OTHER";
            break;
    }
    qDebug();

    qDebug() << "id: " << id;
    qDebug() << "severity: ";
    switch (severity){
        case GL_DEBUG_SEVERITY_LOW:
            qDebug() << "LOW";
            break;
        case GL_DEBUG_SEVERITY_MEDIUM:
            qDebug() << "MEDIUM";
            break;
        case GL_DEBUG_SEVERITY_HIGH:

            qDebug() << "HIGH";
            break;
    }
    qDebug();
    qDebug() << "---------------------opengl-callback-end--------------" ;
}
