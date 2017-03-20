#include <stdio.h>
#include <stdlib.h>
#include "time.h"


#include "material.h"
#include "objects.h"
#include "camera.h"
#include "scene.h"
#include "renderer.h"






int main(int argc, char *argv[]) {

    int samples = 16;            // Default samples per pixel

    if (argc == 2) samples = atoi(argv[1]);
    Camera camera = Camera(glm::dvec3(0, -5, 2.5), glm::dvec3(0,0,0), 1280/2, 720/2);     // Create camera

    Scene scene = Scene();// Create scene

    // Add objects to scene
    scene.add( dynamic_cast<Object*>(new Sphere(glm::dvec3(0,0,-1000), 1000, Material())) );
    scene.add( dynamic_cast<Object*>(new Sphere(glm::dvec3(-1004,0,0), 1000, Material(DIFF, glm::dvec3(0.85,0.4,0.4)))) );
    scene.add( dynamic_cast<Object*>(new Sphere(glm::dvec3(1004,0,0), 1000, Material(DIFF, glm::dvec3(0.4,0.4,0.85)))) );
    scene.add( dynamic_cast<Object*>(new Sphere(glm::dvec3(0,1006,0), 1000, Material())) );
    scene.add( dynamic_cast<Object*>(new Sphere(glm::dvec3(0,0,110), 100, Material(EMIT, glm::dvec3(1,1,1), glm::dvec3(2.2,2.2,2.2)))) );
    scene.add( dynamic_cast<Object*>(new Mesh(glm::dvec3(), "../obj/dragon2.obj", Material(DIFF, glm::dvec3(0.9, 0.9, 0.9)))) );


    Renderer renderer = Renderer(&scene, &camera);
//    renderer.render();
    renderer.runInLoop();
//    renderer.save_image("./render.png");
    return 0;
}