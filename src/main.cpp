/**
* @file      main.cpp
* @brief     Example Boids flocking simulation for CIS 565
* @authors   Liam Boone, Kai Ninomiya, Kangning (Gary) Li
* @date      2013-2017
* @copyright University of Pennsylvania
*/

#include "main.hpp"

// ================
// Configuration
// ================

/**
* C main function.
*/
int main(int argc, char* argv[]) {
  if (init(argc, argv)) {
    mainLoop();
    return 0;
  } else {
    return 1;
  }
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

std::string deviceName;
GLFWwindow *window;

/**
* Initialization of CUDA and GLFW.
*/
bool init(int argc, char **argv) {
    cudaDeviceProp deviceProp;
    int gpuDevice = 0;
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (gpuDevice > device_count) {
        std::cout
            << "Error: GPU device number is greater than the number of devices!"
            << " Perhaps a CUDA-capable GPU is not installed?"
            << std::endl;
        return false;
    }
    cudaGetDeviceProperties(&deviceProp, gpuDevice);
    int major = deviceProp.major;
    int minor = deviceProp.minor;

    std::ostringstream ss;
    ss << "MEGA MINECRAFT " << " [SM " << major << "." << minor << " " << deviceProp.name << "]";
    deviceName = ss.str();

    glfwSetErrorCallback(errorCallback);

    if (!glfwInit()) {
        std::cout
            << "Error: Could not initialize GLFW!"
            << " Perhaps OpenGL 3.3 isn't available?"
            << std::endl;
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(width, height, deviceName.c_str(), NULL, NULL);
    if (!window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mousePositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        return false;
    }

    cudaGLSetGLDevice(0);

    renderer = std::make_unique<Renderer>(window, terrain.get());
    renderer->init();

    initGame();

    return true;
}

void initGame() 
{
    terrain = std::make_unique<Terrain>();
    player = std::make_unique<Player>();
}

void mainLoop() {
    double lastTime = 0;

    while (!glfwWindowShouldClose(window)) 
    {
        glfwPollEvents();

        double time = glfwGetTime();
        double deltaTime = time - lastTime;
        lastTime = time;

        tick();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
}

void tick() {
    player->tick();

    terrain->setCurrentChunkPos(Utils::worldPosToChunkPos(player->getPos()));
    terrain->tick();

    renderer->draw();
}

void errorCallback(int error, const char* description) {
    fprintf(stderr, "error %d: %s\n", error, description);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    //leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    //rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
    //if (leftMousePressed) {
    //    // compute new camera parameters
    //    phi += (xpos - lastX) / width;
    //    theta -= (ypos - lastY) / height;
    //    theta = std::fmax(0.01f, std::fmin(theta, 3.14f));
    //    updateCamera();
    //} else if (rightMousePressed) {
    //    zoom += (ypos - lastY) / height;
    //    zoom = std::fmax(0.1f, std::fmin(zoom, 5.0f));
    //    updateCamera();
    //}

    //lastX = xpos;
    //lastY = ypos;
}