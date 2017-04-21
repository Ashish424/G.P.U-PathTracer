//
// Created by ashish on 4/4/17.
//






#include "BasicScene.hpp"
#include <iostream>
int main(){



    std::ios_base::sync_with_stdio(false);
    //TODO load scene here and send it to app via reference


    BasicScene s(800,600,"Basic Gpu Tracer");
    s.run();





    return 0;
}
