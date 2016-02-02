// Have you remembered to export GLMDIR=/glm?
#include <iostream>
#include <glm/glm.hpp>
#include <SDL.h>
#include "SDLauxiliary.h"
#include "TestModel.h"
#include <limits>

//#define MOVE
#define LIGHT

using namespace std;
using glm::vec3;
using glm::mat3;

/* ----------------------------------------------------------------------------*/
/* GLOBAL VARIABLES                                                            */
vector<Triangle> triangles;

//Use smaller parameters when camera moving for realtime performance
#ifdef MOVE
	const int SCREEN_WIDTH = 150;
	const int SCREEN_HEIGHT = 150;
	float focalLength = 3000.0f;
	vec3 cameraPos(0.0f, 0.0f, -50.0f);
#else
	const int SCREEN_WIDTH = 500;
	const int SCREEN_HEIGHT = 500;
	float focalLength = 150.0f;
	vec3 cameraPos(0.0f, 0.0f, -1.5f);
#endif

mat3 cameraRot = mat3(0.0f);
float yaw = 0.0;

SDL_Surface* screen;
int t;

// omni light variables
vec3 lightPos(0, -0.5f, -0.7f);
vec3 lightColor = 14.0f * vec3(1,1,1);

struct Intersection
{
	vec3 position;
	float distance;
	int triangleIndex;
};

vector<Intersection> closestIntersections;

/* ----------------------------------------------------------------------------*/
/* FUNCTIONS                                                                   */

void Update();
void Draw();
bool ClosestIntersection(vec3 start, vec3 dir, const vector<Triangle>& triangles,
						 Intersection& closestIntersection);
vec3 DirectLight(const Intersection& i);

int main( int argc, char* argv[] )
{
	screen = InitializeSDL( SCREEN_WIDTH, SCREEN_HEIGHT );
	t = SDL_GetTicks();	// Set start value for timer.

	// defines the Cornell Box
	LoadTestModel( triangles );

	//every pixel will have a closest intersection
	size_t i;
	float m = std::numeric_limits<float>::max();
	for(i = 0; i < SCREEN_WIDTH*SCREEN_HEIGHT; i++)
	{
		Intersection intersection;
		intersection.distance = m;
		closestIntersections.push_back(intersection);
			}

	cameraRot[1][1] = 1.0f;

	while( NoQuitMessageSDL() )
	{
		Update();
		Draw();
	}

	SDL_SaveBMP( screen, "screenshot.bmp" );
	return 0;
}


bool ClosestIntersection(vec3 start, vec3 dir, const vector<Triangle>& triangles,
						 Intersection& closestIntersection)
{
	bool intersection = false;
	// check all triangles for intersections
	size_t i;
	for (i = 0; i < triangles.size(); i++)
	{
		// the 3D real vectors that define the triangle
		vec3 v0 = triangles[i].v0;
		vec3 v1 = triangles[i].v1;
		vec3 v2 = triangles[i].v2;

		// edges that are co-planar
		vec3 e1 = v1 - v0;
		vec3 e2 = v2 - v0;
		vec3 b = start - v0;

		// solve linear equation
		mat3 A(-dir, e1, e2);
		vec3 x = glm::inverse(A) * b; // t, u, v

		// checking constraints for point to be in triangle
		float t = x.x, u = x.y, v = x.z;
		//if (u > 0.0f && v > 0.0f && t >= 0.0f && u + v < 1.0f)
		if (u+v <= 1.0f && u >= 0.0f && v >= 0.0f && t >= 0.0f)
		{
			vec3 pos = v0 + (u*e1) + (v*e2);
			float distance = glm::distance(start, pos);
			if (closestIntersection.distance >= distance)
			{
				closestIntersection.position = pos;
				closestIntersection.distance = distance;
				closestIntersection.triangleIndex = i;
			}
			intersection = true;
		}
	}

	return intersection;
}

vec3 DirectLight(const Intersection& i)
{
	// r is distance from lightPos and intersection pos
	float r = glm::distance(i.position, lightPos);
	float A = 4*M_PI*(r*r);
	vec3 P = lightColor;
	// unit vector of direction from surface to light
	vec3 rDir = glm::normalize(lightPos - i.position);
	// unit vector describing normal of surface
	vec3 nDir = glm::normalize(triangles[i.triangleIndex].normal);
	vec3 B = P/A;

	// direct ligth intensity
	vec3 D = B * max(glm::dot(rDir,nDir), 0.0f);

	// diffuse
	// the color stored in the triangle is the reflected fraction of light
	vec3 p = triangles[i.triangleIndex].color;
	vec3 R = p*D;

	return R;
}

void Update()
{
	// Compute frame time:
	int t2 = SDL_GetTicks();
	float dt = float(t2-t);
	t = t2;
	cout << "Render time: " << dt << " ms." << endl;

	// move camera
	vec3 right(cameraRot[0][0], cameraRot[0][1], cameraRot[0][2]);
	vec3 down(cameraRot[1][0], cameraRot[1][1], cameraRot[1][2]);
	vec3 forward(cameraRot[2][0], cameraRot[2][1], cameraRot[2][2]);
	Uint8* keystate = SDL_GetKeyState( 0 );
	if( keystate[SDLK_UP] )
	{
		// Move camera forward
		cameraPos += 1.0f*forward;
	}
	else if( keystate[SDLK_DOWN] )
	{
		// Move camera backward
		cameraPos -= 1.0f*forward;
	}
	if( keystate[SDLK_LEFT] )
	{
		// Move camera to the left
		yaw += 0.01f;
	}
	else if( keystate[SDLK_RIGHT] )
	{
		// Move camera to the right
		yaw -= 0.01f;
	}
	// update rot
	float c = cos(yaw);
	float s = sin(yaw);
	cameraRot[0][0] = c;
	cameraRot[0][2] = s;
	cameraRot[2][0] = -s;
	cameraRot[2][2] = c;

	// moving light
	if (keystate[SDLK_w])
	{
		lightPos += 1.0f*forward;
	}
	else if (keystate[SDLK_s])
	{
		lightPos -= 1.0f*forward;
	}

	// moving light
	if (keystate[SDLK_a])
	{
		lightPos -= 0.1f*right;
	}
	else if (keystate[SDLK_d])
	{
		lightPos += 0.1f*right;
	}

}

void Draw()
{
	if( SDL_MUSTLOCK(screen) )
		SDL_LockSurface(screen);

	// trace a ray for every pixel
	int x, y;
	for (y = 0; y < SCREEN_HEIGHT; y++)
	{
		for (x = 0; x < SCREEN_WIDTH; x++)
		{
			// work out vectors from rotation
			vec3 d(x-SCREEN_WIDTH/2, y - SCREEN_HEIGHT/2, focalLength);
			if ( ClosestIntersection(cameraPos, cameraRot*d, triangles, closestIntersections[y*SCREEN_HEIGHT + x] ))
			{
				// if intersect, use color of closest triangle
				//vec3 color = triangles[closestIntersections[y*SCREEN_HEIGHT+x].triangleIndex].color;
				vec3 color = DirectLight(closestIntersections[y*SCREEN_HEIGHT+x]);
				PutPixelSDL( screen, x, y, color );
			}
			else
			{
				PutPixelSDL( screen, x, y, vec3(0.0f,0.0f,0.0f) );
			}
		}
	}

	if( SDL_MUSTLOCK(screen) )
		SDL_UnlockSurface(screen);

	SDL_UpdateRect( screen, 0, 0, 0, 0 );
}
