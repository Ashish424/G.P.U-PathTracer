//
// Created by ashish on 4/4/17.
//

#include "utilfun.hpp"
#include <cassert>
#include <GLFW/glfw3.h>
#include <fstream>
#include <tiny_obj_loader.h>
#include "cutil_math.h"
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

using std::cout;
using std::endl;
using glm::vec3;
using glm::vec4;
//#define USE_ASSIMP
#define USE_TINY
//#define USE_NONE
namespace uf {
    GLFWwindow *createWindow(int width, int height, const char *title) {

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

        return glfwCreateWindow(width, height, title, nullptr, nullptr);

    }

    int initGlad() {
        return gladLoadGLLoader((GLADloadproc) glfwGetProcAddress);

    }

    GLuint createGlTex2DCuda(GLsizei w, GLsizei h, GLint clampX, GLint clampY, GLint minFilter, GLint magFilter) {
        GLuint texture;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);

// set basic parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

// Create texture data (4-component unsigned byte)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI, w, h, 0, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, NULL);

// Unbind the texture
        glBindTexture(GL_TEXTURE_2D, 0);
        return texture;
    }

    int findCudaDevice() {
        int devID = 0;
        int device_count;
        checkCudaErrors(cudaGetDeviceCount(&device_count));

        if (device_count == 0) {
            cout << "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n" << endl;
            return -1;
        }


        cudaDeviceProp deviceProp;
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

        if (deviceProp.computeMode == cudaComputeModeProhibited) {
            cout << "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice()" <<endl;
            return -1;
        }

        if (deviceProp.major < 1) {
            cout << "gpuDeviceInit(): GPU device does not support CUDA" << endl;
        }

        checkCudaErrors(cudaSetDevice(devID));
        cout << "CUDA Device ID: "<< devID << " " <<  "name: " << deviceProp.name << endl;
        return devID;

    }
    std::vector<char> fileToCharArr(const std::string & filename,bool printFile) {
//FIXME add error checking for filepath
        std::ifstream inputFileHandle;
        inputFileHandle.open(filename);
        auto start = inputFileHandle.tellg();
        inputFileHandle.seekg(0, std::ios::end);
        auto len = inputFileHandle.tellg() - start;
        inputFileHandle.seekg(start);
        std::vector<char> buffer(len);
        inputFileHandle.read(&buffer[0], len);
        //TODO note len!=gcount!!!!
        assert(len>=inputFileHandle.gcount());
        buffer[inputFileHandle.gcount()] = 0;
        inputFileHandle.close();

        if(printFile) {
            std::cout <<"print len is" << len << std::endl;
            for (int i = 0; i < len; ++i) {
                std::cout << buffer[i];
            }
        }
        return buffer;

    }
    GLuint compileShader(const shDf & sd){

        GLuint Shader = glCreateShader(sd.first);
        glShaderSource(Shader, 1, &sd.second, NULL);
        glCompileShader(Shader);
        GLint success;
        GLchar infoLog[512];
        glGetShaderiv(Shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(Shader, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
        }
        return Shader;


    }

    GLuint makeProgram(GLuint vS, GLuint fS,bool deleteDetachShaders){
        GLint success = 0;
        GLchar infoLog[512];
        GLuint sp = glCreateProgram();
        glAttachShader(sp, vS);
        glAttachShader(sp, fS);
        glLinkProgram(sp);
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


    TriMesh loadTris(const char *filename) {


#ifdef USE_ASSIMP
        Assimp::Importer imp;

        TriMesh sendMesh;
        thrust::host_vector<glm::ivec3> temp;
        const aiScene* scene = imp.ReadFile(filename, aiProcess_Triangulate | aiProcess_FlipUVs);
        if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
        {
            cout << "ERROR::ASSIMP::" << imp.GetErrorString() << endl;
            exit(1);
        }
        // Process all the node's meshes (if any)
        for(size_t i = 0; i < scene->mNumMeshes; i++) {
            auto currMesh = scene->mMeshes[i];
            size_t numFaces = currMesh->mNumFaces;
            for (size_t j = 0; j < numFaces; ++j) {

                const aiFace &face = currMesh->mFaces[j];
                //support for only triangular meshes


                assert(face.mNumIndices == 3);
                glm::vec3 currTri[3];
                vec3 off = vec3(0,0,-22);
                for (size_t k = 0; k < 3; ++k) {
                    aiVector3D pos = currMesh->mVertices[face.mIndices[k]];
                    currTri[k] = vec3(pos.x,pos.y,pos.z);
//                    aiVector3D uv = currMesh->mTextureCoords[0][face.mIndices[k]];
//                    aiVector3D normal = currMesh->HasNormals() ? currMesh->mNormals[face.mIndices[k]] : aiVector3D(1.0f, 1.0f, 1.0f);

                    currTri[k]+=off;
                }
                glm::vec4 v04(currTri[0],1);
                glm::vec4 v14(currTri[1]-currTri[0], 0);
                glm::vec4 v24(currTri[2]-currTri[0], 0);
                sendMesh.ve.push_back(v04);
                sendMesh.ve.push_back(v14);
                sendMesh.ve.push_back(v24);

                temp.push_back(glm::ivec3(face.mIndices[0],face.mIndices[1],face.mIndices[2]));

            }

        }



        cout << "vert info" << scene->mMeshes[0]->mNumVertices << endl;
        cout << "face info" << temp.size() << endl;
        for (int l = 0; l < temp.size(); ++l) {
            cout << "printing stuff" << glm::to_string(temp[l])<< endl;
        }





//        for(size_t i = 0;i<sendMesh.ve.size();++i)
//            cout <<glm::to_string(sendMesh.ve[i]) << endl;
        return sendMesh;



#elif defined(USE_NONE)

        TriMesh sendMesh;

        std::ifstream in(filename);

        if (!in.good())
        {
            std::cout << "ERROR: loading obj:(" << filename << ") file not found or not good" << "\n";
//            system("PAUSE");
            exit(0);
        }

        char buffer[256], str[255];
        float f1, f2, f3;

        std::vector<vec3> vertVec;
        std::vector<glm::ivec3> faceVec;

        while (!in.getline(buffer, 255).eof())
        {
            buffer[255] = '\0';
            sscanf(buffer, "%s", str, 255);

            // reading a vertex
            if (buffer[0] == 'v' && (buffer[1] == ' ' || buffer[1] == 32)){
                if (sscanf(buffer, "v %f %f %f", &f1, &f2, &f3) == 3){
                    vertVec.push_back(glm::vec3(f1, f2, f3));
                }
                else{
                    std::cout << "ERROR: vertex not in wanted format in OBJLoader" << "\n";
                    exit(-1);
                }
            }
                // reading faceMtls
            else if (buffer[0] == 'f' && (buffer[1] == ' ' || buffer[1] == 32))
            {
                glm::ivec3 f;
                int nt = sscanf(buffer, "f %d %d %d", &f[0], &f[1], &f[2]);
                if (nt != 3){
                    std::cout << "ERROR: I don't know the format of that FaceMtl" << "\n";
                    exit(-1);
                }

                faceVec.push_back(f);
            }
        }
//        for (size_t i = 0; i < faceVec.size(); i++) {
//            cout << glm::to_string(faceVec[i])<< endl;
//
//        }
//        for (size_t i = 0; i < vertVec.size(); i++) {
//            cout << glm::to_string(vertVec[i])<< endl;
//
//        }



        vec3 off = vec3(0,0,-22);
        for (size_t i = 0; i < faceVec.size(); i++)
        {
            // make a local copy of the triangle vertices
            vec3 v0 = vertVec[faceVec[i][0] - 1];
            vec3 v1 = vertVec[faceVec[i][1] - 1];
            vec3 v2 = vertVec[faceVec[i][2] - 1];

            // translate
            v0 += off;
            v1 += off;
            v2 += off;

            glm::vec4 v04(v0.x, v0.y, v0.z, 0);
            glm::vec4 v14(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z, 0);
            glm::vec4 v24(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z, 0);
            sendMesh.ve.push_back(v04);
            sendMesh.ve.push_back(v14);
            sendMesh.ve.push_back(v24);
        }
        cout <<"num verts loaded " << sendMesh.ve.size() << endl;
        return sendMesh;

#else

        TriMesh sendMesh;
        std::ifstream in(filename);

        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;

        std::string err;
        tinyobj::attrib_t attribs;
        bool ret = tinyobj::LoadObj(&attribs,&shapes, &materials, &err, filename);

        if (!err.empty()) { // `err` may contain warning message.
            std::cout << err << std::endl;
        }

        if (!ret) {
            std::cout << "error in tinyObj"<< std::endl;
            exit(EXIT_FAILURE);
        }

        //TODO disable this debugging info later
        std::cout << "# of shapes    : " << shapes.size() << std::endl;
        std::cout << "# of indexes    : " << shapes[0].mesh.indices.size() << std::endl;
        std::cout << "# of materials : " << materials.size() << std::endl;
        std::cout << "# of verts : " << attribs.vertices.size() << std::endl;


//        for (size_t i = 0; i < attribs.vertices.size()/3; ++i) {
//            cout << attribs.vertices[i] << attribs.vertices[i+1] << attribs.vertices[i+2] << endl;
//
//        }

        for (size_t i = 0; i < shapes.size(); ++i) {
            assert((shapes[i].mesh.indices.size() % 3) == 0);
            for (size_t f = 0; f < shapes[i].mesh.indices.size() / 3; ++f) {

                vec4 currTri[3];
                vec4 currTriNorm[3];

                vec4 off = vec4(0,0,-21,0);
                for(size_t k = 0;k<3;++k){

                    float vX = attribs.vertices[shapes[i].mesh.indices[3*f+k].vertex_index*3];
                    float vY = attribs.vertices[shapes[i].mesh.indices[3*f+k].vertex_index*3+1];
                    float vZ = attribs.vertices[shapes[i].mesh.indices[3*f+k].vertex_index*3+2];

//                    float nX = attribs.normals[shapes[i].mesh.indices[3*f+k].vertex_index*3];
//                    float nY = attribs.normals[shapes[i].mesh.indices[3*f+k].vertex_index*3+1];
//                    float nZ = attribs.normals[shapes[i].mesh.indices[3*f+k].vertex_index*3+2];

                    currTri[k] = vec4(vX,vY,vZ,1);
                    currTri[k]+=off;
//                    currTriNorm[k] = vec4(nX,nY,nZ,0);
                }

                sendMesh.ve.push_back(currTri[0]);
                sendMesh.ve.push_back(currTri[1]-currTri[0]);
                sendMesh.ve.push_back(currTri[2]-currTri[0]);

//                sendMesh.normals.push_back(currTriNorm[0]);
//                sendMesh.normals.push_back(currTriNorm[1]);
//                sendMesh.normals.push_back(currTriNorm[2]);




            }
        }
        return sendMesh;

#endif

    }

    uint64_t hash(uint64_t key) {
        key = (~key) + (key << 21); // key = (key << 21) - key - 1;
        key = key ^ (key >> 24);
        key = (key + (key << 3)) + (key << 8); // key * 265
        key = key ^ (key >> 14);
        key = (key + (key << 2)) + (key << 4); // key * 21
        key = key ^ (key >> 28);
        key = key + (key << 31);
        return key;
    }



    IndexedTriMesh loadIndexedTris(const char *filename){

#ifdef USE_ASSIMP
        IndexedTriMesh sendMesh;
        Assimp::Importer imp;

        const aiScene* scene = imp.ReadFile(std::string(filename),0);
        if(!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
        {
            cout << "ERROR::ASSIMP::" << imp.GetErrorString() << endl;
            exit(1);
        }
        // Process all the node's meshes (if any)
        cout <<"num meshes"<< scene->mNumMeshes << endl;
        for(size_t i = 0; i < scene->mNumMeshes; ++i) {
            auto currMesh = scene->mMeshes[i];
            size_t numFaces = currMesh->mNumFaces;
            for (size_t j = 0; j < numFaces; ++j) {

                const aiFace &face = currMesh->mFaces[j];
                //support for only triangular meshes
                assert(face.mNumIndices == 3);
                sendMesh.triIndexes.push_back(SceneMesh::Triangle(glm::ivec3(face.mIndices[0],face.mIndices[1],face.mIndices[2])));

            }


            for(size_t  m = 0;m<currMesh->mNumVertices;++m){
                aiVector3D pos = currMesh->mVertices[m];
                vec3 off = vec3(0,0,-22);
                sendMesh.ve.push_back(glm::vec3(pos.x,pos.y,pos.z)+off);
                cout << "vertex is is "<<glm::to_string(sendMesh.ve[m]) << endl;
            }

        }

//        cout <<"print verts here" <<scene->mMeshes[0]->mNumVertices << endl;

//        for(int i = 0;i<sendMesh.ve.size();++i){
//            cout << glm::to_string(sendMesh.ve[i])<< endl;
//        }
//        for(int i = 0;i<sendMesh.triIndexes.size();++i){
//            cout <<"index verts " << glm::to_string(sendMesh.triIndexes[i].vertices)<< endl;
//        }




        cout <<"verts num" << sendMesh.ve.size() << endl;
        cout <<"verts num" << sendMesh.triIndexes.size() << endl;

//        cout << "indexes num"<<sendMesh.triIndexes.size() << endl;


        return sendMesh;

#endif




        IndexedTriMesh sendMesh;
        std::ifstream in(filename);

        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;

        std::string err;
        tinyobj::attrib_t attribs;
        bool ret = tinyobj::LoadObj(&attribs,&shapes, &materials, &err, filename);

        if (!err.empty()) { // `err` may contain warning message.
            std::cout << err << std::endl;
        }

        if (!ret) {
            std::cout << "error in tinyObj"<< std::endl;
            exit(EXIT_FAILURE);
        }


        assert(shapes.size() == 1);

        for (size_t f = 0; f < attribs.vertices.size()/3; ++f) {

            //TODO this offset hardcoded remove
            vec3 off = vec3(0,0,-21);
            sendMesh.ve.push_back(glm::vec3(attribs.vertices[0+3*f],attribs.vertices[1+3*f],attribs.vertices[2+3*f])+off);
        }

            for (size_t i = 0; i < shapes.size(); ++i) {
            assert((shapes[i].mesh.indices.size() % 3) == 0);
            for (size_t f = 0; f < shapes[i].mesh.indices.size() / 3; ++f) {

                vec4 currTri[3];
                vec4 currTriNorm[3];


                int in1 = shapes[i].mesh.indices[3*f+0].vertex_index;
                int in2 = shapes[i].mesh.indices[3*f+1].vertex_index;
                int in3 = shapes[i].mesh.indices[3*f+2].vertex_index;

//                    currTriNorm[k] = vec4(nX,nY,nZ,0);

                sendMesh.triIndexes.push_back(SceneMesh::Triangle(glm::ivec3(in1,in2,in3)));




//                sendMesh.normals.push_back(currTriNorm[0]);
//                sendMesh.normals.push_back(currTriNorm[1]);
//                sendMesh.normals.push_back(currTriNorm[2]);




            }
        }
        //TODO disable this debugging info later
        std::cout << "# of shapes    : " << shapes.size() << std::endl;
        std::cout << "# of indexes    : " << sendMesh.triIndexes.size() << std::endl;
        std::cout << "# of verts : " << sendMesh.ve.size()<< std::endl;


        for (int i = 0; i < sendMesh.ve.size(); ++i) {
            cout <<"verts " << glm::to_string(sendMesh.ve[i]) << endl;
        }
        for (int i = 0; i < sendMesh.triIndexes.size(); ++i) {
            cout << "indexes "<< glm::to_string(sendMesh.triIndexes[i].vertices) << endl;
        }



        return sendMesh;





    }

}







