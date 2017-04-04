#include <glad/glad.h>
#include <QtWidgets/QApplication>
#include <QtWidgets/QPushButton>
#include <QtOpenGL/QGLFormat>
#include "Scene.hpp"


int main( int argc, char **argv )
{
    QApplication application(argc,argv);
    QSurfaceFormat format;
    format.setVersion(4,3);
    format.setProfile(QSurfaceFormat::CoreProfile);
    //TODO setup these like glfw parameters
    format.setDepthBufferSize(24);
    format.setStencilBufferSize(8);
    QSurfaceFormat::setDefaultFormat(format);
    Scene c;
    c.show();
    return application.exec();
}