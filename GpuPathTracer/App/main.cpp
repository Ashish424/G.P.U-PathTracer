//
// Created by ashish on 4/4/17.
//

#include "../BasicScene.hpp"
#include <iostream>
int main(){



    std::ios_base::sync_with_stdio(false);

    BasicScene s(640*2,360*2,"Gpu Path Tracer");
    s.run();





    return 0;
}
