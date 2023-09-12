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
#include "glslUtility.hpp"
#include "terrain/terrain.hpp"
#include "player/player.hpp"

//====================================
// GL Stuff
//====================================

int width = 1920;
int height = 1080;

//====================================
// Game things
//====================================

std::unique_ptr<Terrain> terrain;
std::unique_ptr<Player> player;

//====================================
// Setup/init Stuff
//====================================

bool init(int argc, char** argv);
void initGame();

//====================================
// Main
//====================================

int main(int argc, char* argv[]);

//====================================
// Main loop
//====================================

void mainLoop();
void tick();
void errorCallback(int error, const char* description);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
