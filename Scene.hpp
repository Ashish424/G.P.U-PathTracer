//
// Created by ashish on 3/7/17.
//

#ifndef GPUPROJECT_CGCANVAS_HPP
#define GPUPROJECT_CGCANVAS_HPP

#include <QtWidgets/QOpenGLWidget>


class Scene : public QOpenGLWidget
{
Q_OBJECT

public:
    Scene(QWidget *parent=0);
    ~Scene();

    // QOpenGLWidget interface
protected:
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();


private:


};


#endif //GPUPROJECT_CGCANVAS_HPP
