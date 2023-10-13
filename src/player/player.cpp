#include "player.hpp"

#include <glm/gtc/matrix_transform.hpp>

void Player::tick(bool& viewMatChanged)
{
    if (camChanged)
    {
        // TODO recalculate forward/right/up
        float sinTheta = sin(theta);
        float cosTheta = cos(theta);

        forward = glm::vec3(sinTheta, 0, cosTheta);

        viewMat = glm::lookAt(pos, pos + forward, vec3(0, 1, 0));
        viewMatChanged = true;
    }
}

vec3 Player::getPos() 
{
    return pos;
}

vec3 Player::getForward()
{
    return forward;
}

vec3 Player::getRight()
{
    return right;
}

vec3 Player::getUp()
{
    return up;
}

mat4 Player::getViewMat() const
{
    return viewMat;
}

void Player::move(vec3 input)
{
    pos += forward * input.z 
        + right * input.x 
        + vec3(0, input.y, 0);
    camChanged = true;
}

void Player::rotate(float dTheta, float dPhi)
{
    phi = max(-1.565f, min(1.565f, phi + dPhi)); // slightly under pi/2 in both directions
    theta += dTheta;
    camChanged = true;
}