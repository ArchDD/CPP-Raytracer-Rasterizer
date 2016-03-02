/* ----------------------------------------------------------------------------*/
/* ADDITIONAL FEATURES                                                         */
// Cramer's Rule
// Feature Toggling - Can toggle render features at runtime using 1-9 keys
// Supersample Antialiasing (1 key) - An additional N^2 rays are fired per pixel and the resulting colour averaged to smoothen jagged edges
// Soft Shadows (2 key) - A light is split into N lights with 1 / N intensity and a random position jitter added to simulate soft shadows

/* ----------------------------------------------------------------------------*/

#include <iostream>
#include <glm/glm.hpp>
#include <SDL.h>
#include "SDLauxiliary.h"
#include "TestModel.h"
#include <limits>
#include <omp.h>

using namespace std;
using glm::vec3;
using glm::mat3;

/* ----------------------------------------------------------------------------*/
/* GLOBAL VARIABLES                                                            */
vector<Triangle> triangles;

/* RENDER SETTINGS                                                             */
//#define REALTIME

bool MULTITHREADING_ENABLED = false;
int NUM_THREADS = 4; // Set to 0 to get max available

bool AA_ENABLED = false;
int AA_SAMPLES = 3;

bool SOFT_SHADOWS_ENABLED = false;
int SOFT_SHADOWS_SAMPLES = 16;

bool DOF_ENABLED = true;
int DOF_KERNEL_SIZE = 8;
float FOCAL_LENGTH = 0.5f;

/* KEY STATES                                                                  */
// These variables aren't ideal, it'd be better if we could find an SDL function that gives OnKeyDown events rather than
// simply checking if the key is pressed
bool AA_key_pressed = false;
bool shadows_key_pressed = false;
bool DOF_key_pressed = false;

// Use smaller parameters when camera moving for realtime performance
#ifdef REALTIME
	const int SCREEN_WIDTH = 150;
	const int SCREEN_HEIGHT = 150;
	float focalLength = 250.0f;
	vec3 cameraPos(0.0f, 0.0f, -4.3f);
#else
	const int SCREEN_WIDTH = 500;
	const int SCREEN_HEIGHT = 500;
	float focalLength = 250.0f;
	vec3 cameraPos(0.0f, 0.0f, -2.0f);
#endif

mat3 cameraRot = mat3(0.0f);
float yaw = 0.0;

SDL_Surface* screen;
int t;

// Point light variables
vec3 lightPos(0, -0.5f, -0.7f);
vec3 lightColor = 14.0f * vec3(1,1,1);
vec3 indirectLight = 0.2f*vec3(1,1,1);

// Store jittered light positions for soft shadows. Up to 64 samples allowed.
vec3 randomPositions[64];

// Depth of field data containers
float focalDistances[SCREEN_WIDTH * SCREEN_HEIGHT];
vec3 pixelColours[SCREEN_WIDTH * SCREEN_HEIGHT];
vec3 blurredPixels[SCREEN_WIDTH * SCREEN_HEIGHT];

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
						 Intersection& closestIntersection, bool isLight, int x, int y);
vec3 DirectLight(const Intersection& i);
float RandomNumber();
void CalculateDOF();

int main( int argc, char* argv[] )
{
	screen = InitializeSDL( SCREEN_WIDTH, SCREEN_HEIGHT );

	// Request as many threads as the system can provide
	if(NUM_THREADS == 0) 
		NUM_THREADS = omp_get_max_threads();

    omp_set_num_threads(NUM_THREADS);

    if(MULTITHREADING_ENABLED)
    	cout << "Multithreading enabled with " << NUM_THREADS << " threads" << endl;
	if(AA_ENABLED)
		cout << "Antialiasing enabled with samples: " << AA_SAMPLES << endl;
	if(SOFT_SHADOWS_ENABLED)
		cout << "Soft Shadows enabled with samples: " << SOFT_SHADOWS_SAMPLES << endl;

	// Set start value for timer
	t = SDL_GetTicks();

	// Generate the Cornell Box
	LoadTestModel( triangles );

	// Every pixel will have a closest intersection
	size_t i;
	float m = std::numeric_limits<float>::max();

	for(i = 0; i < SCREEN_WIDTH*SCREEN_HEIGHT; i++)
	{
		Intersection intersection;
		intersection.distance = m;
		closestIntersections.push_back(intersection);
	}

	cameraRot[1][1] = 1.0f;

	// Generate jittered light positions
	for(i = 0; i < SOFT_SHADOWS_SAMPLES; i++)
	{
		vec3 randomPos(lightPos.x + (RandomNumber() * 0.1f), lightPos.y + (RandomNumber() * 0.1f), lightPos.z + (RandomNumber() * 0.1f));
		randomPositions[i] = randomPos;
	}

	while( NoQuitMessageSDL() )
	{
		Update();
		Draw();
		CalculateDOF();
		// Reset intersection distances
		for(i = 0; i < SCREEN_WIDTH*SCREEN_HEIGHT; i++)
		{
			closestIntersections[i].distance = m;
		}
	}

	SDL_SaveBMP( screen, "screenshot.bmp" );

	return 0;
}


bool ClosestIntersection(vec3 start, vec3 dir, const vector<Triangle>& triangles,
						 Intersection& closestIntersection, bool isLight, int x, int y)
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

		// keep this for speed comparison
		//mat3 A(-dir, e1, e2);vec3 x = glm::inverse(A) * b; float t = x.x, u = x.y, v = x.z;

		// Cramer's rule: valid when there is a solution
		//anticommutative
		vec3 e1e2 = glm::cross(e1,e2);
		vec3 be2 = glm::cross(b,e2);
		vec3 e1b = glm::cross(e1,b);

		vec3 negD = -dir;

		float e1e2b = e1e2.x*b.x+e1e2.y*b.y+e1e2.z*b.z;
		float e1e2d = e1e2.x*negD.x+e1e2.y*negD.y+e1e2.z*negD.z;
		float be2d =  be2.x*negD.x+be2.y*negD.y+be2.z*negD.z;
		float e1bd =  e1b.x*negD.x+e1b.y*negD.y+e1b.z*negD.z;

		// checking constraints for point to be in triangle
		float t = e1e2b/e1e2d, u = be2d/e1e2d, v = e1bd/e1e2d;

		if (u+v <= 1.0f && u >= 0.0f && v >= 0.0f && t >= 0.0f)
		{
			vec3 pos = v0 + (u*e1) + (v*e2);
			float distance = glm::distance(start, pos);
			if (closestIntersection.distance >= distance)
			{
				closestIntersection.position = pos;
				closestIntersection.distance = distance;
				closestIntersection.triangleIndex = i;
				if(!isLight) 
					focalDistances[y*SCREEN_HEIGHT + x] = distance - FOCAL_LENGTH;
			}
			intersection = true;
		}
	}

	return intersection;
}

// Returns a random number between -0.5 and 0.5
float RandomNumber()
{
	return ((double) rand() / (RAND_MAX)) - 0.5f;
}

vec3 DirectLight(const Intersection& i)
{
	int counter;
	int samples;
	vec3 result(0.0f,0.0f,0.0f);

	if(SOFT_SHADOWS_ENABLED)
		samples = SOFT_SHADOWS_SAMPLES;
	else
		samples = 1;

	for(counter = 0; counter < samples; counter++)
	{
		vec3 position;

		if(samples != 1)
		{
			position = randomPositions[counter];
		}
		else
		{
			position = lightPos;
		}

		// r is distance from lightPos and intersection pos
		float r = glm::distance(i.position, position);
		float A = 4*M_PI*(r*r);
		vec3 P = lightColor;
		// unit vector of direction from surface to light
		vec3 rDir = glm::normalize(position - i.position);
		// unit vector describing normal of surface
		vec3 nDir = glm::normalize(triangles[i.triangleIndex].normal);
		vec3 B = P/A;

		// direct light intensity
		vec3 D = B * max(glm::dot(rDir,nDir), 0.0f);

		// direct shadows
		Intersection j;
		j.distance = std::numeric_limits<float>::max();
		// to avoid comparing with self, trace from light and reverse direction
		if (ClosestIntersection(position, -rDir, triangles, j, true, 0, 0))
		{
			// if intersection is closer to light source than self
			if (j.distance < r*0.99f) // small multiplier to reduce noise
				D = vec3 (0.0f, 0.0f, 0.0f);
		}

		// diffuse
		// the color stored in the triangle is the reflected fraction of light
		result += D;
	}
	
	result /= (float)samples;
	vec3 p = triangles[i.triangleIndex].color;
	return result*p;
}

void Update()
{
	// Compute frame time
	int t2 = SDL_GetTicks();
	float dt = float(t2-t);
	t = t2;
	cout << "Render time: " << dt << " ms." << endl;

	// Adjust camera transform
	vec3 right(cameraRot[0][0], cameraRot[0][1], cameraRot[0][2]);
	vec3 down(cameraRot[1][0], cameraRot[1][1], cameraRot[1][2]);
	vec3 forward(cameraRot[2][0], cameraRot[2][1], cameraRot[2][2]);
	Uint8* keystate = SDL_GetKeyState( 0 );
	
	if( keystate[SDLK_UP] )
	{
		// Move camera forward
		cameraPos += 0.1f*forward;
	}
	else if( keystate[SDLK_DOWN] )
	{
		// Move camera backward
		cameraPos -= 0.1f*forward;
	}
	if( keystate[SDLK_LEFT] )
	{
		// Rotate camera to the left
		yaw += 0.1f;
	}
	else if( keystate[SDLK_RIGHT] )
	{
		// Rotate camera to the right
		yaw -= 0.1f;
	}

	// Update camera rotation matrix
	float c = cos(yaw);
	float s = sin(yaw);
	cameraRot[0][0] = c;
	cameraRot[0][2] = s;
	cameraRot[2][0] = -s;
	cameraRot[2][2] = c;
	

	// Light movement controls
	if (keystate[SDLK_w])
	{
		lightPos += 0.1f*forward;
		for(int i = 0; i < SOFT_SHADOWS_SAMPLES; i++)
		{
			randomPositions[i] += 0.1f*forward;
		}
	}
	else if (keystate[SDLK_s])
	{
		lightPos -= 0.1f*forward;
		for(int i = 0; i < SOFT_SHADOWS_SAMPLES; i++)
		{
			randomPositions[i] -= 0.1f*forward;
		}
	}

	// Light movement controls
	if (keystate[SDLK_a])
	{
		lightPos -= 0.1f*right;
		for(int i = 0; i < SOFT_SHADOWS_SAMPLES; i++)
		{
			randomPositions[i] -= 0.1f*right;
		}
	}
	else if (keystate[SDLK_d])
	{
		lightPos += 0.1f*right;
		for(int i = 0; i < SOFT_SHADOWS_SAMPLES; i++)
		{
			randomPositions[i] += 0.1f*right;
		}
	}

	// Need to check if key has been released to stop the option toggling every frame
	if(!AA_key_pressed && keystate[SDLK_1])
	{
		AA_ENABLED = !AA_ENABLED;
		cout << "Antialiasing toggled to " << AA_ENABLED << endl;
		AA_key_pressed = true;
	}
	else if (!keystate[SDLK_1])
		AA_key_pressed = false;

	if(!shadows_key_pressed && keystate[SDLK_2])
	{
		SOFT_SHADOWS_ENABLED = !SOFT_SHADOWS_ENABLED;
		cout << "Soft Shadows toggled to " << SOFT_SHADOWS_ENABLED << endl;
		shadows_key_pressed = true;
	}
	else if (!keystate[SDLK_2])
		shadows_key_pressed = false;

	if(!DOF_key_pressed && keystate[SDLK_3])
	{
		DOF_ENABLED = !DOF_ENABLED;
		cout << "Depth of Field toggled to " << DOF_ENABLED << endl;
		DOF_key_pressed = true;
	}
	else if (!keystate[SDLK_3])
		DOF_key_pressed = false;

	if (keystate[SDLK_z])
	{
		FOCAL_LENGTH += 0.1f;
		cout << "Focal length is " << FOCAL_LENGTH << endl;
	}
		
	if (keystate[SDLK_x])
	{
		FOCAL_LENGTH -= 0.1f;
		cout << "Focal length is " << FOCAL_LENGTH << endl;
	}
		

}

void Draw()
{

	// trace a ray for every pixel
	int x, y, z, z2;
	int realSamples; // Number of AA samples to use. Set to 1 if AA is disabled
	float x1, y1;

	if(AA_ENABLED)
		realSamples = AA_SAMPLES;
	else
		realSamples = 1;

	for (y = 0; y < SCREEN_HEIGHT; y++)
	{
		for (x = 0; x < SCREEN_WIDTH; x++)
		{
			vec3 avgColor(0.0f,0.0f,0.0f);
			if(realSamples > 1) 
				y1 = y - 0.5f;
			else
				y1 = y;

			for(z = 0; z < realSamples; z++)
			{
				if(realSamples > 1) 
					x1 = x - 0.5f;
				else
					x1 = x;

				for(z2 = 0; z2 < realSamples; z2++)
				{
					// work out vectors from rotation
					vec3 d(x1-(float)SCREEN_WIDTH/2.0f, y1 - (float)SCREEN_HEIGHT/2.0f, focalLength);
					if ( ClosestIntersection(cameraPos, cameraRot*d, triangles, closestIntersections[y*SCREEN_HEIGHT + x], false, x, y ))
					{
						// if intersect, use color of closest triangle
						vec3 color = DirectLight(closestIntersections[y*SCREEN_HEIGHT+x]);
						vec3 D = color;
						vec3 N = indirectLight;
						vec3 T = D + N;
						vec3 p = triangles[closestIntersections[y*SCREEN_HEIGHT+x].triangleIndex].color;
						vec3 R = p*T;

						// direct shadows cast to point from light
						avgColor += R;

						x1 += (1.0f / (float) (realSamples - 1));
					}
				}
				y1 += (1.0f / (float) (realSamples - 1));
			}

			avgColor /= (float)(realSamples * realSamples);
			pixelColours[y*SCREEN_HEIGHT + x] = avgColor;

		}
	}
}

void CalculateDOF()
{
	if( SDL_MUSTLOCK(screen) )
		SDL_LockSurface(screen);

	float totalPixels = DOF_KERNEL_SIZE * DOF_KERNEL_SIZE;

	for (int y = 1; y < SCREEN_HEIGHT - 1; y++)
	{
		for (int x = 1; x < SCREEN_WIDTH - 1; x++)
		{
			vec3 finalColour(0.0f,0.0f,0.0f);
			for(int z = floor(DOF_KERNEL_SIZE / -2.0f); z < ceil(DOF_KERNEL_SIZE / 2.0f); z++)
			{
				for(int z2 = floor(DOF_KERNEL_SIZE / -2.0f); z2 < ceil(DOF_KERNEL_SIZE / 2.0f); z2++)
				{
					float weighting;
					if(z == 0 && z2 == 0)
						weighting = max((1 - focalDistances[y*SCREEN_HEIGHT+x]) * totalPixels, focalDistances[y*SCREEN_HEIGHT+x]);
					else
						weighting = min(abs(focalDistances[y*SCREEN_HEIGHT+x]), 1.0f);


					finalColour += pixelColours[(y+z)*SCREEN_HEIGHT+(x+z2)] * weighting;
				}
			}
			finalColour /= totalPixels;
			PutPixelSDL( screen, x, y, finalColour );
		}
	}

	if( SDL_MUSTLOCK(screen) )
		SDL_UnlockSurface(screen);
		

	SDL_UpdateRect( screen, 0, 0, 0, 0 );

}
