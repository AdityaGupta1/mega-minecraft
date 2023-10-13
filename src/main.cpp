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

    // TODO handle window resize

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        return false;
    }

    cudaGLSetGLDevice(0);

    initGame();

    renderer = std::make_unique<Renderer>(window, terrain.get(), player.get());
    renderer->init();

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

        tick(deltaTime);
    }

    glfwDestroyWindow(window);
    glfwTerminate();
}

void errorCallback(int error, const char* description) {
    fprintf(stderr, "error %d: %s\n", error, description);
}

glm::ivec3 playerMovement = glm::ivec3(0);
glm::vec3 playerMovementSensitivity = glm::vec3(2.f, 1.2f, 2.f);

int actionToInt(int action)
{
    switch (action)
    {
    case GLFW_PRESS:
        return 1;
    case GLFW_RELEASE:
        return -1;
    default:
        return 0;
    }
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) 
{
    switch (key)
    {
    case GLFW_KEY_ESCAPE:
        if (action == GLFW_PRESS)
        {
            glfwSetWindowShouldClose(window, GL_TRUE);
            return;
        }
        break;
    case GLFW_KEY_W:
        playerMovement.z += actionToInt(action);
        break;
    case GLFW_KEY_S:
        playerMovement.z -= actionToInt(action);
        break;
    case GLFW_KEY_A:
        playerMovement.x += actionToInt(action);
        break;
    case GLFW_KEY_D:
        playerMovement.x -= actionToInt(action);
        break;
    case GLFW_KEY_SPACE:
    case GLFW_KEY_E:
        playerMovement.y += actionToInt(action);
        break;
    case GLFW_KEY_LEFT_SHIFT:
    case GLFW_KEY_Q:
        playerMovement.y -= actionToInt(action);
        break;
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    //leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    //rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
}

const float mouseSensitivity = -0.002f;

void mousePositionCallback(GLFWwindow* window, double mouseX, double mouseY) {
    int width, height;
    glfwGetWindowSize(window, &width, &height);
    float centerX = width / 2;
    float centerY = height / 2;

    float dTheta = (mouseX - centerX) * mouseSensitivity;
    float dPhi = (mouseY - centerY) * mouseSensitivity;

    if (dTheta != 0 || dPhi != 0)
    {
        player->rotate(dTheta, dPhi);
    }

    glfwSetCursorPos(window, centerX, centerY);
}

void tick(float deltaTime)
{
    if (playerMovement.x != 0 || playerMovement.y != 0 || playerMovement.z != 0)
    {
        player->move(glm::vec3(playerMovement) * playerMovementSensitivity * deltaTime);
    }
    bool viewMatChanged;
    player->tick(viewMatChanged);

    terrain->setCurrentChunkPos(Utils::worldPosToChunkPos(player->getPos()));
    terrain->tick();

    renderer->draw(viewMatChanged);
}