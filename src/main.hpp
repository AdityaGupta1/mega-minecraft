#pragma once

#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "terrain/terrain.hpp"
#include "player/player.hpp"
#include "rendering/optixRenderer.hpp"
#include "rendering/renderer.hpp"

//====================================
// GL Stuff
//====================================

glm::ivec2 windowSize = ivec2(1920, 1080);
//glm::ivec2 windowSize = ivec2(1920 / 2, 1080 / 2);
bool windowSizeChanged;

std::unique_ptr<OptixRenderer> optix;
std::unique_ptr<Renderer> renderer;

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
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void framebufferSizeCallback(GLFWwindow* window, int width, int height);
