//
// Created by ashish on 3/7/17.
//

#ifndef GPUPROJECT_CGCANVAS_HPP
#define GPUPROJECT_CGCANVAS_HPP

#include <glad/glad.h>
#include <QtWidgets/QOpenGLWidget>


class CGCanvas : public QOpenGLWidget
{
Q_OBJECT

public:
    CGCanvas(QWidget *parent=0);
    ~CGCanvas();

    // QOpenGLWidget interface
protected:
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();



private:


};


#endif //GPUPROJECT_CGCANVAS_HPP
