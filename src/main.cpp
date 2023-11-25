#include "main.hpp"

#include "terrain/block.hpp"
#include "defines.hpp"

#define DEBUG_START_IN_FREE_CAM_MODE 1

int main(int argc, char* argv[]) {
  if (init(argc, argv)) {
    mainLoop();
    return 0;
  } else {
    return 1;
  }
}

std::string deviceName;
GLFWwindow *window;

bool init(int argc, char **argv) {
    cudaDeviceProp deviceProp;
    int gpuDevice = 0;
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    printf("num devices: %d\n", device_count);
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
    ss << "Mega Minecraft";
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

    window = glfwCreateWindow(windowSize.x, windowSize.y, deviceName.c_str(), NULL, NULL);
    if (!window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mousePositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN); 
    
    glfwSetCursorPos(window, windowSize.x / 2.f, windowSize.y / 2.f);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        return false;
    }

    cudaGLSetGLDevice(0);

    BlockUtils::init();
    BiomeUtils::init();

    constructTerrainAndPlayer();

#if DEBUG_USE_GL_RENDERER
    renderer = std::make_unique<Renderer>(window, &windowSize, terrain.get(), player.get());
    if (!renderer->init())
    {
        return false;
    }
#else
    optix = std::make_unique<OptixRenderer>(window, &windowSize, terrain.get(), player.get());
    terrain->setOptixRenderer(optix.get());
#endif

    terrain->init(); // call after creating CUDA context in OptixRenderer

    return true;
}

void constructTerrainAndPlayer()
{
    terrain = std::make_unique<Terrain>();
    player = std::make_unique<Player>();
}

void mainLoop() {
    double lastTime = 0;
    double fpsLastTime = 0;
    int frames = 0;

    while (!glfwWindowShouldClose(window)) 
    {
        glfwPollEvents();

        double time = glfwGetTime();
        double deltaTime = time - lastTime;
        lastTime = time;

        ++frames;
        if (time - fpsLastTime > 1.f)
        {
            float fps = frames / (time - fpsLastTime);
            fpsLastTime = time;
            frames = 0;

            std::ostringstream ss;
            ss << "[";
            ss.precision(1);
            ss << std::fixed << fps;
            ss << " fps] " << deviceName;
            glfwSetWindowTitle(window, ss.str().c_str());
        }

        tick(deltaTime);
    }

    glfwDestroyWindow(window);
    glfwTerminate();
}

void errorCallback(int error, const char* description) {
    fprintf(stderr, "error %d: %s\n", error, description);
}

glm::ivec3 playerMovement = glm::ivec3(0);
glm::vec3 playerMovementSensitivity = glm::vec3(10.0f, 8.0f, 10.0f);
bool shiftPressed = false;
bool altPressed = false;

#if DEBUG_START_IN_FREE_CAM_MODE
bool freeCam = true;
#else
bool freeCam = false;
#endif

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
    case GLFW_KEY_Q:
        playerMovement.y -= actionToInt(action);
        break;
    case GLFW_KEY_Z:
        if (action == GLFW_RELEASE)
            player->toggleCamMode();
        break;
    case GLFW_KEY_RIGHT:
        if (action == GLFW_PRESS || action == GLFW_REPEAT)
            player->rotate(-0.1f, 0, true);
        break;
    case GLFW_KEY_LEFT:
        if (action == GLFW_PRESS || action == GLFW_REPEAT)
            player->rotate(0.1f, 0, true);
        break;
    case GLFW_KEY_UP:
        if (action == GLFW_PRESS || action == GLFW_REPEAT)
            player->rotate(0, 0.1f, true);
        break;
    case GLFW_KEY_DOWN:
        if (action == GLFW_PRESS || action == GLFW_REPEAT)
            player->rotate(0, -0.1f, true);
        break;
    case GLFW_KEY_LEFT_SHIFT:
        if (action == GLFW_PRESS)
        {
            shiftPressed = true;
        }
        else if (action == GLFW_RELEASE)
        {
            shiftPressed = false;
        }
        break;
    case GLFW_KEY_LEFT_ALT:
        if (action == GLFW_PRESS)
        {
            altPressed = true;
        }
        else if (action == GLFW_RELEASE)
        {
            altPressed = false;
        }
        break;
    case GLFW_KEY_C:
#if DEBUG_USE_GL_RENDERER
        if (action == GLFW_PRESS)
        {
            renderer->setZoomed(true);
        }
        else if (action == GLFW_RELEASE)
        {
            renderer->setZoomed(false);
        }
#else
        if (action == GLFW_PRESS)
        {
            optix->setZoomed(true);
        }
        else if (action == GLFW_RELEASE)
        {
            optix->setZoomed(false);
        }
#endif
        break;
    case GLFW_KEY_P:
        if (action == GLFW_RELEASE)
        {
#if DEBUG_USE_GL_RENDERER
            renderer->toggleTimePaused();
#else
            optix->toggleTimePaused();
#endif
        }
        break;
    case GLFW_KEY_O:
        if (action == GLFW_RELEASE)
        {
            const vec3 playerPos = player->getPos();
            terrain->debugPrintCurrentChunkInfo(vec2(playerPos.x, playerPos.z));
        }
        break;
    case GLFW_KEY_V:
        if (action == GLFW_RELEASE)
        {
            const vec3 playerPos = player->getPos();
            terrain->debugPrintCurrentZoneInfo(vec2(playerPos.x, playerPos.z));
        }
        break;
    case GLFW_KEY_L:
        if (action == GLFW_RELEASE)
        {
            const vec3 playerPos = player->getPos();
            terrain->debugPrintCurrentColumnLayers(vec2(playerPos.x, playerPos.z));
        }
        break;
    case GLFW_KEY_X:
        if (action == GLFW_RELEASE)
        {
            const vec3 playerPos = player->getPos();
            terrain->debugForceGatherHeightfield(vec2(playerPos.x, playerPos.z));
        }
        break;
    case GLFW_KEY_F:
        if (action == GLFW_RELEASE)
        {
            freeCam = !freeCam;
        }
        break;
    case GLFW_KEY_K:
        if (action == GLFW_PRESS)
        {
            auto playerPos = player->getPos();
            printf("player position: (%.2f, %.2f, %.2f)\n", playerPos.x, playerPos.y, playerPos.z);
        }
        break;
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    //leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    //rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
}

const float mouseSensitivity = -0.0025f;

void mousePositionCallback(GLFWwindow* window, double mouseX, double mouseY) {
    float centerX = windowSize.x / 2.f;
    float centerY = windowSize.y / 2.f;

    float dTheta = (mouseX - centerX) * mouseSensitivity;
    float dPhi = (mouseY - centerY) * mouseSensitivity;

    if (dTheta != 0 || dPhi != 0)
    {
        player->rotate(dTheta, dPhi, false);
    }

    glfwSetCursorPos(window, centerX, centerY);
}

void framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    windowSize = glm::ivec2(width, height);
    windowSizeChanged = true;
}

void tick(float deltaTime)
{
    if (playerMovement.x != 0 || playerMovement.y != 0 || playerMovement.z != 0)
    {
        glm::vec3 playerMovementNormalized;
        if (playerMovement.x != 0 || playerMovement.z != 0)
        {
            glm::vec2 playerMovementHorizontal = glm::vec2(playerMovement.x, playerMovement.z);
            playerMovementHorizontal = glm::normalize(playerMovementHorizontal) * glm::length(playerMovementHorizontal);
            playerMovementNormalized = glm::vec3(playerMovementHorizontal.x, playerMovement.y, playerMovementHorizontal.y);
        }
        else
        {
            playerMovementNormalized = playerMovement;
        }

        float playerMovementMultiplier = 1.f;
        if (shiftPressed)
        {
            playerMovementMultiplier *= 8.f;

            if (altPressed)
            {
                playerMovementMultiplier *= 4.f;
            }
        }
        else if (altPressed)
        {
            playerMovementMultiplier *= 0.25f;
        }

        player->move(glm::vec3(playerMovementNormalized) * playerMovementSensitivity * playerMovementMultiplier * deltaTime);
    }
    bool viewMatChanged = false;
    player->tick(&viewMatChanged);

#if !DEBUG_USE_GL_RENDERER
    terrain->destroyFarChunkVbos();
#endif

    if (!freeCam)
    {
        terrain->setCurrentChunkPos(Utils::worldPosToChunkPos(player->getPos()));
    }
    terrain->tick(deltaTime);

#if DEBUG_USE_GL_RENDERER
    renderer->draw(deltaTime, viewMatChanged, windowSizeChanged);
#else
    if (viewMatChanged) {
        optix->setCamera();
    }
    optix->render(deltaTime);
#endif

    windowSizeChanged = false;
}