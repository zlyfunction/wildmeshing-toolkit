#include <igl/readOBJ.h>
#include <igl/opengl/glfw/Viewer.h>
#include <iostream>
  Eigen::MatrixXd V, uv;
  Eigen::MatrixXi F;

// This function is called every time a keyboard button is pressed
bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{
  std::cout<<"Key: "<<key<<" "<<(unsigned int)key<<std::endl;
  if (key == '1')
  {
    // Clear should be called before drawing the mesh
    viewer.data().clear();
    // Draw_mesh creates or updates the vertices and faces of the displayed mesh.
    // If a mesh is already displayed, draw_mesh returns an error if the given V and
    // F have size different than the current ones
    viewer.data().set_mesh(V, F);
    viewer.core().align_camera_center(V,F);
  }
  else if (key == '2')
  {
    viewer.data().clear();
    viewer.data().set_mesh(uv, F);
    viewer.core().align_camera_center(uv,F);
  }

  return false;
}


int main(int argc, char *argv[])
{

  igl::readOBJ(argv[1], V, uv, uv, F, F, F);

  igl::opengl::glfw::Viewer viewer;
  // Register a keyboard callback that allows to switch between
  // the two loaded meshes
  viewer.callback_key_down = &key_down;
  viewer.data().set_mesh(V, F);
  viewer.launch();
}