// Have you remembered to export GLMDIR=/glm?
#include <iostream>
#include <glm/glm.hpp>
#include <SDL.h>
#include "SDLauxiliary.h"
#include "TestModel.h"
#include <limits>

using namespace std;
using glm::vec3;
using glm::mat3;

/* ----------------------------------------------------------------------------*/
/* GLOBAL VARIABLES                                                            */
vector<Triangle> triangles;
const int SCREEN_WIDTH = 500;
const int SCREEN_HEIGHT = 500;
SDL_Surface* screen;
int t;

/*float focalLength = 50.0;
vec3 cameraPos(0.0, 0.0, -1.0);*/
float focalLength = 50.0;
vec3 cameraPos(0.0, 0.0, -1.0);


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
		if (u > 0 && v > 0 && t >= 0 && u + v < 1)
		{
			//vec3 pos(u+v0.x,v+v0.y,v0.z);
			vec3 pos = v0 + (u*e1) + (v*e2);
			float distance = glm::distance(start, pos);
			if (closestIntersection.distance > distance)
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

void Update()
{
	// Compute frame time:
	int t2 = SDL_GetTicks();
	float dt = float(t2-t);
	t = t2;
	cout << "Render time: " << dt << " ms." << endl;
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
			vec3 d(x-SCREEN_WIDTH/2, y - SCREEN_HEIGHT/2, focalLength);
			if ( ClosestIntersection(cameraPos, d, triangles, closestIntersections[y*SCREEN_HEIGHT + x] ))
			{
				// if intersect, use color of closest triangle
				vec3 color = triangles[closestIntersections[y*SCREEN_HEIGHT+x].triangleIndex].color;
				PutPixelSDL( screen, x, y, color );
			}
			else
			{
				PutPixelSDL( screen, x, y, vec3(0,0,0) );
			}
		}
	}

	if( SDL_MUSTLOCK(screen) )
		SDL_UnlockSurface(screen);

	SDL_UpdateRect( screen, 0, 0, 0, 0 );
}
