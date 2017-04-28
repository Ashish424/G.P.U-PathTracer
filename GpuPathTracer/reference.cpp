
#include <cstdlib>
#include <iostream>
#include <glad/glad.h>

#include <imgui.h>
#include <imgui_internal.h>
#include <GLFW/glfw3.h>
#include <imgui_impl_glfw_gl3.h>

#define MY_DEBUG_OPENGL_APP


GLFWwindow * setupGlfw(int width, int height, const std::string &title);
void initShaderAndBuffers();
void restoreState();


void APIENTRY openglCallbackFunction(GLenum source,
                                     GLenum type,
                                     GLuint id,
                                     GLenum severity,
                                     GLsizei length,
                                     const GLchar* message,
                                     const void* userParam);

const GLchar* vertexShaderSource = "#version 330 core\n"
        "layout (location = 0) in vec3 position;\n"
        "void main()\n"
        "{\n"
        "gl_Position = vec4(position.x, position.y, position.z, 1.0);\n"
        "}\0";
const GLchar* fragmentShaderSource = "#version 330 core\n"
        "out vec4 color;\n"
        "void main()\n"
        "{\n"
        "color = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
        "}\n\0";

GLuint VBO, VAO;
GLuint shaderProgram;
int main(int argc, char ** argv) {



    using namespace std;

    int screenWidth = 800,screenHeight = 600;
    auto window = setupGlfw(screenWidth,screenHeight, "MouseTest");
    // Setup ImGui binding
    ImGui_ImplGlfwGL3_Init(window, true);


    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress))
    {
        cout << "Failed to initialize OpenGL context" << endl;
        return -1;
    }

    //ready to make GL Calls
    glViewport(-400,-400, screenWidth, screenHeight);




#ifdef MY_DEBUG_OPENGL_APP
    const GLubyte* renderer = glGetString(GL_RENDERER);
    const GLubyte* version  = glGetString(GL_VERSION);
    std::cout << "Renderer :" << renderer << std::endl;
    std::cout << "Version :" << version<<std::endl;






    if(glDebugMessageCallback){
        std::cout << "Register OpenGL debug callback " << std::endl;
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
        std::cout << "glDebugMessageCallback not available" << std::endl;
#endif







//    GLboolean ans = GL_FALSE;
//    glGetBooleanv(GL_CULL_FACE,&ans);
//    printf("culling is %d\n",ans);


    initShaderAndBuffers();
    //setup initial OpenGL state
    restoreState();


    bool show_test_window = true;
    bool show_another_window = true;




    while (!glfwWindowShouldClose(window)) {
        // Check if any events have been activated (key pressed, mouse moved etc.) and call corresponding response functions
        glfwPollEvents();
        //Drawing
        {
            glClearColor(211 / 256.0f, 211 / 256.0f, 211 / 256.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);



            //draw your stuff
            glUseProgram(shaderProgram);
            glBindVertexArray(VAO);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            glBindVertexArray(0);
        }
        //GUI
        {
            ImGui_ImplGlfwGL3_NewFrame();

            // 1. Show a simple window
            // Tip: if we don't call ImGui::Begin()/ImGui::End() the widgets appears in a window automatically called "Debug"
            {
//                static float f = 0.0f;
//                ImGui::Text("Hello, world!");
//                ImGui::SliderFloat("float", &f, 0.0f, 1.0f);
////                ImGui::ColorEdit3("clear color", (float *) &clear_color);
//                if (ImGui::Button("Test Window")) show_test_window ^= 1;
//                if (ImGui::Button("Another Window")) show_another_window ^= 1;
//                ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
//                            ImGui::GetIO().Framerate);
//                if(ImGui::Button("MyButton")){
//                    cout <<"print stuff here" << endl;
//                }
            }

            // 2. Show another simple window, this time using an explicit Begin/End pair
            if (show_another_window) {
                ImGui::SetNextWindowPos(ImVec2(400, 200));
                ImGui::SetNextWindowSize(ImVec2(200, 100), ImGuiSetCond_FirstUseEver);
                ImGui::Begin("Test Window", &show_another_window);
                float f;
                ImGui::SliderFloat("float", &f, 0.0f, 1.0f);

                ImGui::Text("Hello");
                if (ImGui::Button("Button")) { cout <<"button named button clicked" << endl;
                }
                ImGui::End();
            }

            // 3. Show the ImGui test window. Most of the sample code is in ImGui::ShowTestWindow()
//            if (show_test_window) {
//                ImGui::SetNextWindowPos(ImVec2(650, 20), ImGuiSetCond_FirstUseEver);
//                ImGui::ShowTestWindow(&show_test_window);

//
//            }



            // Rendering
            int display_w, display_h;
            glfwGetFramebufferSize(window, &display_w, &display_h);
            glViewport(0, 0, display_w, display_h);

            ImGui::Render();

        }

        glfwSwapBuffers(window);


    }





    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);


    ImGui_ImplGlfwGL3_Shutdown();
    glfwTerminate();

    return 0;



}



void restoreState(){

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glViewport(rand()%10,rand()%10,800,600);



//    glEnable(GL_CULL_FACE);
//    glFrontFace(GL_CW);
//    glCullFace(GL_FRONT);



}

GLFWwindow * setupGlfw(int width, int height, const std::string &title) {
    glfwInit();

    glfwSetTime(0);

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
#ifdef MY_DEBUG_OPENGL_APP
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT,GLFW_TRUE);
#endif


    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

//
// Create a GLFWwindow object
    GLFWwindow *window = glfwCreateWindow(width,height,title.c_str(), nullptr, nullptr);
    if (window == nullptr) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return nullptr;
    }
    glfwMakeContextCurrent(window);
    return window;
}



void initShaderAndBuffers() {



    GLfloat vertices[] = {
            0.0f, 0.5f, 0.49f,  // Top Right
            0.5f, -0.5f, 0.49f,  // Bottom Right
            -0.5f, -0.5f, 0.49f,  // Bottom Left

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
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
// Fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
// Check for compile time errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
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
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
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


    glBindVertexArray(0); // Unbind VAO (it's always a good thing to unbind any buffer/array to prevent strange bugs), remember: do NOT unbind the EBO, keep it bound to this VAO





}
void APIENTRY openglCallbackFunction(GLenum source,
                                     GLenum type,
                                     GLuint id,
                                     GLenum severity,
                                     GLsizei length,
                                     const GLchar* message,
                                     const void* userParam){

//using namespace std;
//cout << "---------------------opengl-callback-start------------" << endl;
//cout << "message: "<< message << endl;
//cout << "type: ";
//switch (type) {
//case GL_DEBUG_TYPE_ERROR:
//cout << "ERROR";
//break;
//case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
//cout << "DEPRECATED_BEHAVIOR";
//break;
//case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
//cout << "UNDEFINED_BEHAVIOR";
//break;
//case GL_DEBUG_TYPE_PORTABILITY:
//cout << "PORTABILITY";
//break;
//case GL_DEBUG_TYPE_PERFORMANCE:
//cout << "PERFORMANCE";
//break;
//case GL_DEBUG_TYPE_OTHER:
//cout << "OTHER";
//break;
//}
//cout << endl;
//
//cout << "id: " << id << endl;
//cout << "severity: ";
//switch (severity){
//case GL_DEBUG_SEVERITY_LOW:
//cout << "LOW";
//break;
//case GL_DEBUG_SEVERITY_MEDIUM:
//cout << "MEDIUM";
//break;
//case GL_DEBUG_SEVERITY_HIGH:
//cout << "HIGH";
//break;
//}
//cout << endl;
//cout << "---------------------opengl-callback-end--------------" << endl;
}