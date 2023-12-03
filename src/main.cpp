#include "main.hpp"

#include "terrain/block.hpp"

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
    int gpuDevice = GPU_DEVICE;
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

    CUDA_CHECK(cudaSetDevice(gpuDevice));
    // CUDA_CHECK(cudaGLSetGLDevice(1));
    cudaGetDeviceProperties(&deviceProp, gpuDevice);
    int major = deviceProp.major;
    int minor = deviceProp.minor;

    std::ostringstream ss;
    ss << "Mega Minecraft";
    deviceName = ss.str();
#if USE_D3D11_RENDERER
    initD3DWindow();
    timer = std::make_unique<StepTimer>();
#else
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
#endif

    BlockUtils::init();
    BiomeUtils::init();

    constructTerrainAndPlayer();

#if DEBUG_USE_GL_RENDERER
    renderer = std::make_unique<Renderer>(window, &windowSize, terrain.get(), player.get());
    if (!renderer->init())
    {
        return false;
    }
#elif USE_D3D11_RENDERER
    d3dRenderer = std::make_unique<D3D11Renderer>(g_hWnd, &windowSize.x, &windowSize.y);
    optixRenderer = std::make_unique<OptixRenderer>(d3dRenderer.get(), &windowSize, terrain.get(), player.get());
#else
    optixRenderer = std::make_unique<OptixRenderer>(window, &windowSize, terrain.get(), player.get());
#endif
    terrain->setOptixRenderer(optixRenderer.get());
    
    terrain->init(); // call after creating CUDA context in OptixRenderer

    return true;
}

void constructTerrainAndPlayer()
{
    terrain = std::make_unique<Terrain>();
    player = std::make_unique<Player>();
}

void mainLoop() {
#if USE_D3D11_RENDERER
    timer->ResetElapsedTime();

    MSG msg = { 0 };
    while (WM_QUIT != msg.message)
    {
        if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        else
        {
            bool newFPS;
            uint32_t fps = timer->GetFramesPerSecond(&newFPS);
            if (newFPS) {
                std::ostringstream ss;
                ss << "[" << fps << " fps] " << deviceName;
                SetWindowText(g_hWnd, ss.str().c_str());
            }
            timer->Tick();
            tick(timer->GetElapsedSeconds());
        }
    }
#else
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
#endif
}

void errorCallback(int error, const char* description) {
    fprintf(stderr, "error %d: %s\n", error, description);
}

glm::ivec3 playerMovement = glm::ivec3(0);
glm::vec3 playerMovementSensitivity = glm::vec3(10.0f, 8.0f, 10.0f);
bool shiftPressed = false;
bool altPressed = false;
bool trackInput = false;

#if DEBUG_START_IN_FREE_CAM_MODE
bool freeCam = true;
#else
bool freeCam = false;
#endif

#if USE_D3D11_RENDERER
void initD3DWindow()
{
    WNDCLASSEX wcex;
    wcex.cbSize = sizeof(WNDCLASSEX);
    wcex.style = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc = MsgProc;
    wcex.cbClsExtra = 0;
    wcex.cbWndExtra = 0;
    wcex.hInstance = NULL;
    wcex.hIcon = NULL;
    wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wcex.lpszMenuName = nullptr;
    wcex.lpszClassName = "Mega Minecraft";
    wcex.hIconSm = NULL;
    RegisterClassEx(&wcex);

    g_hWnd = CreateWindow(wcex.lpszClassName, "Mega Minecraft", WS_OVERLAPPEDWINDOW,
        0, 0, windowSize.x + 2 * GetSystemMetrics(SM_CXSIZEFRAME), windowSize.y + 2 * GetSystemMetrics(SM_CYSIZEFRAME) + GetSystemMetrics(SM_CYMENU),
        NULL, NULL, wcex.hInstance, NULL);

    ShowWindow(g_hWnd, SW_SHOWMAXIMIZED);
    UpdateWindow(g_hWnd);
}

LRESULT CALLBACK MsgProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
    case WM_PAINT:
        ValidateRect(hWnd, NULL);
        break;

    case WM_DESTROY:
        PostQuitMessage(0);
        break;

    case WM_ACTIVATE:
        if (LOWORD(wParam) != WA_INACTIVE) {
            SetCapture(g_hWnd);
            ShowCursor(FALSE);
            trackInput = true;
        }
        else {
            ReleaseCapture();
            // ShowCursor(TRUE);
            trackInput = false;
        }
        break;

    case WM_KILLFOCUS:
        ReleaseCapture();
        ShowCursor(TRUE);
        trackInput = false;
        break;

    case WM_SIZE:
        if (wParam != SIZE_MINIMIZED) {
            UINT newWidth = LOWORD(lParam);
            UINT newHeight = HIWORD(lParam);
            windowSize.x = newWidth;
            windowSize.y = newHeight;
            if (d3dRenderer)
                d3dRenderer->onResize();
            if (optixRenderer)
                optixRenderer->onResize();
        }
        break;

    case WM_KEYDOWN:
    case WM_KEYUP:
        if (trackInput)
            keyCallback(hWnd, message, wParam, lParam);
        break;

    case WM_MOUSEMOVE:
        if (trackInput)
            mousePositionCallback(hWnd, message, wParam, lParam);
        break;

    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }

    return 0;
}

int actionToInt(UINT action)
{
    switch (action)
    {
    case WM_KEYDOWN:
        return 1;
    case WM_KEYUP:
        return 0;
    default:
        return 0;
    }
}

void keyCallback(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    shiftPressed = (GetAsyncKeyState(VK_LSHIFT) & 0x8000) != 0;
    altPressed = (GetAsyncKeyState(VK_LMENU) & 0x8000) != 0;
    switch (wParam)
    {
    case VK_ESCAPE:
        PostQuitMessage(0);
        break;
    case 'W':
        playerMovement.z = actionToInt(message);
        break;
    case 'S':
        playerMovement.z = -actionToInt(message);
        break;
    case 'A':
        playerMovement.x = actionToInt(message);
        break;
    case 'D':
        playerMovement.x = -actionToInt(message);
        break;
    case VK_SPACE:
    case 'E':
        playerMovement.y = actionToInt(message);
        break;
    case 'Q':
        playerMovement.y = -actionToInt(message);
        break;
    case 'Z':
        if (message == WM_KEYUP)
            player->toggleCamMode();
        break;
    case VK_RIGHT:
        if (message == WM_KEYDOWN)
            player->rotate(-0.1f, 0, true);
        break;
    case VK_LEFT:
        if (message == WM_KEYDOWN)
            player->rotate(0.1f, 0, true);
        break;
    case VK_UP:
        if (message == WM_KEYDOWN)
            player->rotate(0, 0.1f, true);
        break;
    case VK_DOWN:
        if (message == WM_KEYDOWN)
            player->rotate(0, -0.1f, true);
        break;
    case VK_LSHIFT:
        if (message == WM_KEYDOWN)
        {
            shiftPressed = true;
        }
        else if (message == WM_KEYUP)
        {
            shiftPressed = false;
        }
        break;
    case VK_LMENU:
        if (message == WM_KEYDOWN)
        {
            altPressed = true;
        }
        else if (message == WM_KEYUP)
        {
            altPressed = false;
        }
        break;
    case 'C':
#if DEBUG_USE_GL_RENDERER
        if (message == WM_KEYDOWN)
        {
            renderer->setZoomed(true);
        }
        else if (message == WM_KEYUP)
        {
            renderer->setZoomed(false);
        }
#else
        if (message == WM_KEYDOWN)
        {
            optixRenderer->setZoomed(true);
        }
        else if (message == WM_KEYUP)
        {
            optixRenderer->setZoomed(false);
        }
#endif
        break;
    case 'P':
        if (message == WM_KEYUP)
        {
#if DEBUG_USE_GL_RENDERER
            renderer->toggleTimePaused();
#else
            optixRenderer->toggleTimePaused();
#endif
        }
        break;
    case 'O':
        if (message == WM_KEYUP)
        {
            const vec3 playerPos = player->getPos();
            terrain->debugPrintCurrentChunkInfo(vec2(playerPos.x, playerPos.z));
        }
        break;
    case 'V':
        if (message == WM_KEYUP)
        {
            const vec3 playerPos = player->getPos();
            terrain->debugPrintCurrentZoneInfo(vec2(playerPos.x, playerPos.z));
        }
        break;
    case 'L':
        if (message == WM_KEYUP)
        {
            const vec3 playerPos = player->getPos();
            terrain->debugPrintCurrentColumnLayers(vec2(playerPos.x, playerPos.z));
        }
        break;
    case 'X':
        if (message == WM_KEYUP)
        {
            const vec3 playerPos = player->getPos();
            terrain->debugForceGatherHeightfield(vec2(playerPos.x, playerPos.z));
        }
        break;
    case 'F':
        if (message == WM_KEYUP)
        {
            freeCam = !freeCam;
        }
        break;
    case 'K':
        if (message == WM_KEYDOWN)
        {
            auto playerPos = player->getPos();
            printf("player position: (%.2f, %.2f, %.2f)\n", playerPos.x, playerPos.y, playerPos.z);
        }
        break;
#if !DEBUG_USE_GL_RENDERER
    case VK_OEM_4: // [
        if (message == WM_KEYUP)
        {
            optixRenderer->addTime(-5.f);
        }
        break;
    case VK_OEM_6: // ]
        if (message == WM_KEYUP)
        {
            optixRenderer->addTime(5.f);
        }
        break;
#endif
    }
}

const float mouseSensitivity = -0.0025f;
const float movementThreshold = 0.002f;

void mousePositionCallback(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    float centerX = windowSize.x / 2.f;
    float centerY = windowSize.y / 2.f;
    POINT centerPoint = { centerX, centerY };
    ClientToScreen(g_hWnd, &centerPoint);

    int mouseX = GET_X_LPARAM(lParam);
    int mouseY = GET_Y_LPARAM(lParam);

    float dTheta = (mouseX - centerX) * mouseSensitivity;
    float dPhi = (mouseY - centerY) * mouseSensitivity;

    if (abs(dTheta) > movementThreshold || abs(dPhi) > movementThreshold)
    {
        player->rotate(dTheta, dPhi, false);
    }

    SetCursorPos(centerPoint.x, centerPoint.y);
}

#else

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
            optixRenderer->setZoomed(true);
        }
        else if (action == GLFW_RELEASE)
        {
            optixRenderer->setZoomed(false);
        }
#endif
        break;
    case GLFW_KEY_P:
        if (action == GLFW_RELEASE)
        {
#if DEBUG_USE_GL_RENDERER
            renderer->toggleTimePaused();
#else
            optixRenderer->toggleSunPaused();
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
#endif

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
        optixRenderer->setCamera();
    }
    optixRenderer->render(deltaTime);
#endif

    windowSizeChanged = false;
}