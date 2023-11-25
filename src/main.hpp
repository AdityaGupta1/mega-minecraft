#pragma once

#include "defines.hpp"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <exception>
#include <string>
#include <sstream>
#include <cuda_runtime.h>

#if USE_D3D11_RENDERER
#include <windows.h>
#include <windowsx.h>
#include <d3d11_1.h>
#include <cuda_d3d11_interop.h>
#else
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#endif

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "terrain/terrain.hpp"
#include "player/player.hpp"
#include "rendering/optixRenderer.hpp"
#include "rendering/renderer.hpp"
#include "rendering/d3d11Renderer.h"

//
// Adapted from StepTimer.h by Chuck Walbourn
//
class StepTimer
{
public:
    StepTimer() noexcept(false) :
        m_elapsedTicks(0),
        m_totalTicks(0),
        m_leftOverTicks(0),
        m_frameCount(0),
        m_framesPerSecond(0),
        m_framesThisSecond(0),
        m_qpcSecondCounter(0)
    {
        if (!QueryPerformanceFrequency(&m_qpcFrequency))
        {
            throw std::exception();
        }

        if (!QueryPerformanceCounter(&m_qpcLastTime))
        {
            throw std::exception();
        }

        // Initialize max delta to 1/10 of a second.
        m_qpcMaxDelta = static_cast<uint64_t>(m_qpcFrequency.QuadPart / 10);
    }

    // Get elapsed time since the previous Update call.
    uint64_t GetElapsedTicks() const noexcept { return m_elapsedTicks; }
    double GetElapsedSeconds() const noexcept { return TicksToSeconds(m_elapsedTicks); }

    // Get total time since the start of the program.
    uint64_t GetTotalTicks() const noexcept { return m_totalTicks; }
    double GetTotalSeconds() const noexcept { return TicksToSeconds(m_totalTicks); }

    // Get total number of updates since start of the program.
    uint32_t GetFrameCount() const noexcept { return m_frameCount; }

    // Get the current framerate.
    uint32_t GetFramesPerSecond(bool* hasUpdate) noexcept { *hasUpdate = m_hasFPSUpdate; m_hasFPSUpdate = false; return m_framesPerSecond; }

    // Integer format represents time using 10,000,000 ticks per second.
    static constexpr uint64_t TicksPerSecond = 10000000;

    static constexpr double TicksToSeconds(uint64_t ticks) noexcept { return static_cast<double>(ticks) / TicksPerSecond; }
    static constexpr uint64_t SecondsToTicks(double seconds) noexcept { return static_cast<uint64_t>(seconds * TicksPerSecond); }

    // After an intentional timing discontinuity (for instance a blocking IO operation)
    // call this to avoid having the fixed timestep logic attempt a set of catch-up
    // Update calls.

    void ResetElapsedTime()
    {
        if (!QueryPerformanceCounter(&m_qpcLastTime))
        {
            throw std::exception();
        }

        m_leftOverTicks = 0;
        m_framesPerSecond = 0;
        m_framesThisSecond = 0;
        m_qpcSecondCounter = 0;
    }

    // Update timer state
    void Tick()
    {
        // Query the current time.
        LARGE_INTEGER currentTime;

        if (!QueryPerformanceCounter(&currentTime))
        {
            throw std::exception();
        }

        uint64_t timeDelta = static_cast<uint64_t>(currentTime.QuadPart - m_qpcLastTime.QuadPart);

        m_qpcLastTime = currentTime;
        m_qpcSecondCounter += timeDelta;

        // Clamp excessively large time deltas (e.g. after paused in the debugger).
        if (timeDelta > m_qpcMaxDelta)
        {
            timeDelta = m_qpcMaxDelta;
        }

        // Convert QPC units into a canonical tick format. This cannot overflow due to the previous clamp.
        timeDelta *= TicksPerSecond;
        timeDelta /= static_cast<uint64_t>(m_qpcFrequency.QuadPart);

        const uint32_t lastFrameCount = m_frameCount;

        // Variable timestep update logic.
        m_elapsedTicks = timeDelta;
        m_totalTicks += timeDelta;
        m_leftOverTicks = 0;
        m_frameCount++;

        // Track the current framerate.
        if (m_frameCount != lastFrameCount)
        {
            m_framesThisSecond++;
        }

        if (m_qpcSecondCounter >= static_cast<uint64_t>(m_qpcFrequency.QuadPart))
        {
            m_hasFPSUpdate = true;
            m_framesPerSecond = m_framesThisSecond;
            m_framesThisSecond = 0;
            m_qpcSecondCounter %= static_cast<uint64_t>(m_qpcFrequency.QuadPart);
        }
    }

private:
    // Source timing data uses QPC units.
    LARGE_INTEGER m_qpcFrequency;
    LARGE_INTEGER m_qpcLastTime;
    uint64_t m_qpcMaxDelta;

    // Derived timing data uses a canonical tick format.
    uint64_t m_elapsedTicks;
    uint64_t m_totalTicks;
    uint64_t m_leftOverTicks;

    // Members for tracking the framerate.
    uint32_t m_frameCount;
    uint32_t m_framesPerSecond;
    uint32_t m_framesThisSecond;
    uint64_t m_qpcSecondCounter;
    bool m_hasFPSUpdate;
};

//====================================
// D3D11 Stuff
//====================================
#if USE_D3D11_RENDERER
HWND g_hWnd;

std::unique_ptr<D3D11Renderer> d3dRenderer;
#endif
//====================================
// Render Stuff
//====================================

glm::uvec2 windowSize = uvec2(1920, 1080);
//glm::ivec2 windowSize = ivec2(1920 / 2, 1080 / 2);
bool windowSizeChanged;

std::unique_ptr<OptixRenderer> optix;
std::unique_ptr<Renderer> renderer;

std::unique_ptr<StepTimer> timer;

//====================================
// Game things
//====================================

std::unique_ptr<Terrain> terrain;
std::unique_ptr<Player> player;

//====================================
// Setup/init Stuff
//====================================

bool init(int argc, char** argv);
void constructTerrainAndPlayer();

//====================================
// Main
//====================================

int main(int argc, char* argv[]);

//====================================
// Main loop
//====================================

void mainLoop();
void tick(float deltaTime);
void errorCallback(int error, const char* description);
#if USE_D3D11_RENDERER
void initD3DWindow();
LRESULT CALLBACK MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
void keyCallback(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
// void mouseButtonCallback(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
void mousePositionCallback(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
#else
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void framebufferSizeCallback(GLFWwindow* window, int width, int height);
#endif