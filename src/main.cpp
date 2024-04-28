#include <iostream>
#include <filesystem>
#include "renderer.hpp"

double x = 0, y = 0;
bool isPressed = false, isWPressed = false;
void inputCheck(GLFWwindow* _window, renderer_t& _renderer) {
	if (glfwGetMouseButton(_window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
		double _x, _y;
		glfwGetCursorPos(_window, &_x, &_y);
		if (!isPressed) {
			x = _x; y = _y;
			isPressed = true;
			return;
		}
		double dx = x - _x, dy = y - _y;
		_renderer.rotX += dx * 360 / _renderer.winSizeX;
		if (_renderer.rotX >= 360) _renderer.rotX -= 360;
		_renderer.rotYZ += dy * 360 / _renderer.winSizeY;
		if (_renderer.rotYZ >= 360) _renderer.rotYZ -= 360;

		isPressed = true;
		x = _x; y = _y;
	}
	else {
		isPressed = false;
	}

	return;
}

void scroll_callback(GLFWwindow* _window, double _xoffset, double _yoffset) {
	renderer_t* _r = (renderer_t*)glfwGetWindowUserPointer(_window);
	if (_yoffset < 0) {
		_r->zoom *= 1.1f;
	}
	else {
		_r->zoom *= 0.9f;
		_r->zoom = std::max(_r->zoom, 1.0f);
	}

	(void)_xoffset;
	return;
}

int32_t main(int32_t _argc, char** _argv) {
	if (_argc < 2) {
		printf("Incorrect calling method!\nShould be: ./obj_inspect_inator.exe <filepath>\n");
		return EXIT_FAILURE;
	}
	else if (!std::filesystem::exists(_argv[1])) {
		printf("Couldn't find requested file! Closing!\n");
		return EXIT_FAILURE;
	}

	renderer_t renderer;
	renderer.initRenderer(_argv[1]);
	glfwSetScrollCallback(renderer.window, scroll_callback);

	while (!glfwWindowShouldClose(renderer.window)) {
		glfwPollEvents();
		renderer.drawFrame();
		inputCheck(renderer.window, renderer);
	}

	renderer.cleanup();
	return EXIT_SUCCESS;
}