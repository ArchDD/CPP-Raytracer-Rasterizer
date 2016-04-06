/* ----------------------------------------------------------------------------*/
/* ADDITIONAL FEATURES                                                         */
// Cramer's Rule
// Feature Toggling - Can toggle render features and settings at runtime
// Modular lighting system (2 to generate random light, 3 to delete newest light). Allows for multiple lights, properties defined in Light class in TestModel.h
// Multithreading (4 to toggle, 5-6 to change number of threads) - Uses OpenMP to do calculations across multiple threads
// Supersample Antialiasing (7 key) - An additional N^2 rays are fired per pixel and the resulting colour averaged to smoothen jagged edges
// Soft Shadows (8 key) - A light is split into N lights with 1 / N intensity and a random position jitter added to simulate soft shadows
// Depth of Field (9 to toggle, [ and ] to change focal length) - Distance vectors relative to focal length stored for each pixel, 
// used to set neighbour weightings in blur kernel

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

bool MULTITHREADING_ENABLED = true;
int NUM_THREADS; // Set by code
int SAVED_THREADS; // Stores thread value when changed

bool AA_ENABLED = false;
int AA_SAMPLES = 3;

bool SOFT_SHADOWS_ENABLED = false;
int SOFT_SHADOWS_SAMPLES = 16;

bool DOF_ENABLED = false;
int DOF_KERNEL_SIZE = 8;
float FOCAL_LENGTH = 1.3f;

int NUM_LIGHTS = 0;
Light lights[32];

/* KEY STATES                                                                  */
bool AA_key_pressed = false;
bool shadows_key_pressed = false;
bool DOF_key_pressed = false;
bool OMP_key_pressed = false;
bool thread_add_key_pressed = false;
bool thread_subtract_key_pressed = false;
bool delete_light_key_pressed = false;
bool add_light_key_pressed = false;

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
bool isUpdated = true;

// Ambient Lighting
vec3 indirectLight = 0.2f*vec3(1,1,1);

// Store jittered light positions for soft shadows. Needs minimum size of num lights * soft shadow samples.
vec3 randomPositions[256];

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
void AddLight(vec3 position, vec3 color, float intensity);
void DeleteLight();

int main( int argc, char* argv[] )
{
	screen = InitializeSDL( SCREEN_WIDTH, SCREEN_HEIGHT );
	AddLight(vec3(0, -0.5f, -0.7f), vec3(1,1,1), 14 );

	// Request as many threads as the system can provide
	NUM_THREADS = omp_get_max_threads();
    omp_set_num_threads(NUM_THREADS);

    // Set NUM_THREADS to how many the system can actually provide
    #pragma omp parallel
    {
    	int ID = omp_get_thread_num();
    	if(ID == 0)
    		NUM_THREADS = omp_get_num_threads();
    	    SAVED_THREADS = NUM_THREADS;
    }

    if(MULTITHREADING_ENABLED)
    {
    	cout << "Multithreading enabled with " << NUM_THREADS << " threads" << endl;
    }
    else
    	omp_set_num_threads(1);
    	
	if(AA_ENABLED)
		cout << "Antialiasing enabled with samples: " << AA_SAMPLES << endl;
	if(SOFT_SHADOWS_ENABLED)
		cout << "Soft Shadows enabled with samples: " << SOFT_SHADOWS_SAMPLES << endl;
	if(DOF_ENABLED)
		cout << "DoF enabled with kernel size: " << DOF_KERNEL_SIZE << endl;

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


	while( NoQuitMessageSDL() )
	{
		Update();
		if (isUpdated)
		{
			Draw();
			isUpdated = false;
		}
	}

	SDL_SaveBMP( screen, "screenshot.bmp" );

	return 0;
}

void AddLight(vec3 position, vec3 color, float intensity)
{
	lights[NUM_LIGHTS].position = position;
	lights[NUM_LIGHTS].color = color;
	lights[NUM_LIGHTS].intensity = intensity;

	for(int i = 0; i < SOFT_SHADOWS_SAMPLES; i++)
	{
		vec3 randomPos(lights[NUM_LIGHTS].position.x + (RandomNumber() * 0.08f), lights[NUM_LIGHTS].position.y + (RandomNumber() * 0.08f), lights[NUM_LIGHTS].position.z + (RandomNumber() * 0.08f));
		randomPositions[(NUM_LIGHTS*SOFT_SHADOWS_SAMPLES) + i] = randomPos;
	}

	NUM_LIGHTS++;
}

void DeleteLight()
{
	if(NUM_LIGHTS > 0)
		NUM_LIGHTS--;
}


bool ClosestIntersection(vec3 start, vec3 dir, const vector<Triangle>& triangles,
						 Intersection& closestIntersection, bool isLight, int x, int y)
{
	bool intersection = false;
	// check all triangles for intersections

	for (size_t i = 0; i < triangles.size(); i++)
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
	vec3 result2(0,0,0);

	if(SOFT_SHADOWS_ENABLED)
		samples = SOFT_SHADOWS_SAMPLES;
	else
		samples = 1;

	for(int k = 0; k < NUM_LIGHTS; k++)
	{
		for(counter = 0; counter < samples; counter++)
		{
			vec3 position;
			vec3 lightColor = lights[k].color * lights[k].intensity;

			if(samples != 1)
			{
				position = randomPositions[(k*SOFT_SHADOWS_SAMPLES) + counter];
			}
			else
			{
				position = lights[k].position;
			}

			// r is distance from lightPos and intersection pos
			float r = glm::distance(i.position, position);
			float A = 4*M_PI*(r*r);
			vec3 P = lightColor /= (float) samples;
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
		//result /= (float) samples;
		result2 += result;
	}

	vec3 p = triangles[i.triangleIndex].color;
	return result2*p;
}

void Update()
{
	// Compute frame time
	int t2 = SDL_GetTicks();

	// Reset intersection distances
	float m = std::numeric_limits<float>::max();
	for(int i = 0; i < SCREEN_WIDTH*SCREEN_HEIGHT; i++)
	{
		closestIntersections[i].distance = m;
	}

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
		isUpdated = true;
	}
	else if( keystate[SDLK_DOWN] )
	{
		// Move camera backward
		cameraPos -= 0.1f*forward;
		isUpdated = true;
	}
	if( keystate[SDLK_LEFT] )
	{
		// Rotate camera to the left
		yaw += 0.1f;
		isUpdated = true;
	}
	else if( keystate[SDLK_RIGHT] )
	{
		// Rotate camera to the right
		yaw -= 0.1f;
		isUpdated = true;
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
		lights[0].position += 0.1f*forward;
		for(int i = 0; i < SOFT_SHADOWS_SAMPLES; i++)
		{
			randomPositions[i] += 0.1f*forward;
		}
		isUpdated = true;
	}
	else if (keystate[SDLK_s])
	{
		lights[0].position -= 0.1f*forward;
		for(int i = 0; i < SOFT_SHADOWS_SAMPLES; i++)
		{
			randomPositions[i] -= 0.1f*forward;
		}
		isUpdated = true;
	}

	// Light movement controls
	if (keystate[SDLK_a])
	{
		lights[0].position -= 0.1f*right;
		for(int i = 0; i < SOFT_SHADOWS_SAMPLES; i++)
		{
			randomPositions[i] -= 0.1f*right;
		}
		isUpdated = true;
	}
	else if (keystate[SDLK_d])
	{
		lights[0].position += 0.1f*right;
		for(int i = 0; i < SOFT_SHADOWS_SAMPLES; i++)
		{
			randomPositions[i] += 0.1f*right;
		}
		isUpdated = true;
	}

	// Need to check if key has been released to stop the option toggling every frame
	if(!AA_key_pressed && keystate[SDLK_7])
	{
		AA_ENABLED = !AA_ENABLED;
		cout << "Antialiasing toggled to " << AA_ENABLED << endl;
		AA_key_pressed = true;
		isUpdated = true;
	}
	else if (!keystate[SDLK_7])
	{
		AA_key_pressed = false;
	}

	if(!shadows_key_pressed && keystate[SDLK_8])
	{
		SOFT_SHADOWS_ENABLED = !SOFT_SHADOWS_ENABLED;
		cout << "Soft Shadows toggled to " << SOFT_SHADOWS_ENABLED << endl;
		shadows_key_pressed = true;
		isUpdated = true;
	}
	else if (!keystate[SDLK_8])
	{
		shadows_key_pressed = false;
	}

	if(!DOF_key_pressed && keystate[SDLK_9])
	{
		DOF_ENABLED = !DOF_ENABLED;
		cout << "Depth of Field toggled to " << DOF_ENABLED << endl;
		DOF_key_pressed = true;
		isUpdated = true;
	}
	else if (!keystate[SDLK_9])
	{
		DOF_key_pressed = false;
	}

	if (keystate[SDLK_RIGHTBRACKET])
	{
		FOCAL_LENGTH += 0.1f;
		cout << "Focal length is " << FOCAL_LENGTH << endl;
		isUpdated = true;
	}
	else if (keystate[SDLK_LEFTBRACKET])
	{
		FOCAL_LENGTH -= 0.1f;
		cout << "Focal length is " << FOCAL_LENGTH << endl;
		isUpdated = true;
	}

	if(!OMP_key_pressed && keystate[SDLK_4])
	{
		MULTITHREADING_ENABLED = !MULTITHREADING_ENABLED;
		if(!MULTITHREADING_ENABLED)
			NUM_THREADS = 1;
		else
			NUM_THREADS = SAVED_THREADS;
		omp_set_num_threads(NUM_THREADS);
		cout << "Multithreading toggled to " << MULTITHREADING_ENABLED << endl;
		OMP_key_pressed = true;
		isUpdated = true;
	}
	else if (!keystate[SDLK_4])
	{
		OMP_key_pressed = false;
	}

	if(!thread_add_key_pressed && keystate[SDLK_6])
	{
		NUM_THREADS++;
		SAVED_THREADS = NUM_THREADS;
		omp_set_num_threads(NUM_THREADS);
		cout << "Threads increased to " << NUM_THREADS << endl;
		thread_add_key_pressed = true;
		isUpdated = true;
	}
	else if (!keystate[SDLK_6])
	{
		thread_add_key_pressed = false;
	}

	if(!thread_subtract_key_pressed && keystate[SDLK_5])
	{
		NUM_THREADS--;
		SAVED_THREADS = NUM_THREADS;
		omp_set_num_threads(NUM_THREADS);
		cout << "Threads decreased to " << NUM_THREADS << endl;
		thread_subtract_key_pressed = true;
		isUpdated = true;
	}
	else if (!keystate[SDLK_5])
	{
		thread_subtract_key_pressed = false;
	}

	if(!add_light_key_pressed && keystate[SDLK_2])
	{
		AddLight(vec3(RandomNumber() * 2.0f, RandomNumber() * 2.0f, RandomNumber() * 2.0f),vec3(abs(RandomNumber()) * 2.0f + 0.2f,abs(RandomNumber()) * 2.0f + 0.2f,abs(RandomNumber()) * 2.0f + 0.2f),abs(RandomNumber()) * 20.0f);
		cout << "Spawned a light" << endl;
		add_light_key_pressed = true;
		isUpdated = true;
	}
	else if (!keystate[SDLK_2])
	{
		add_light_key_pressed = false;
	}

	if(!delete_light_key_pressed && keystate[SDLK_3])
	{
		DeleteLight();
		cout << "Deleted a light" << endl;
		delete_light_key_pressed = true;
		isUpdated = true;
	}
	else if (!keystate[SDLK_3])
	{
		delete_light_key_pressed = false;
	}
		

}

void Draw()
{
	int realSamples; // Number of AA samples to use. Set to 1 if AA is disabled

	if(AA_ENABLED)
		realSamples = AA_SAMPLES;
	else
		realSamples = 1;

	// This is the loop that needs parallelisation
	#pragma omp parallel for schedule(auto)
	for (int y = 0; y < SCREEN_HEIGHT; y++)
	{
		float x1, y1;
		for (int x = 0; x < SCREEN_WIDTH; x++)
		{
			vec3 avgColor(0.0f,0.0f,0.0f);
			if(realSamples > 1) 
				y1 = y - 0.5f;
			else
				y1 = y;

			for(int z = 0; z < realSamples; z++)
			{
				if(realSamples > 1) 
					x1 = x - 0.5f;
				else
					x1 = x;

				for(int z2 = 0; z2 < realSamples; z2++)
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

	CalculateDOF();
}

void CalculateDOF()
{
	if( SDL_MUSTLOCK(screen) )
		SDL_LockSurface(screen);

	// Total number of pixels in the kernel
	float totalPixels = DOF_KERNEL_SIZE * DOF_KERNEL_SIZE;

	// Spawn threads
	#pragma omp parallel for schedule(auto)
	for (int y = 1; y < SCREEN_HEIGHT - 1; y++)
	{
		for (int x = 1; x < SCREEN_WIDTH - 1; x++)
		{
			vec3 finalColour(0.0f,0.0f,0.0f);
			if(DOF_ENABLED)
			{
				// Start from top left of kernel
				for(int z = ceil(DOF_KERNEL_SIZE / -2.0f); z < ceil(DOF_KERNEL_SIZE / 2.0f); z++)
				{
					for(int z2 = ceil(DOF_KERNEL_SIZE / -2.0f); z2 < ceil(DOF_KERNEL_SIZE / 2.0f); z2++)
					{
						float weighting;
						if(z == 0 && z2 == 0)
							weighting = 1 - (min(abs(focalDistances[y*SCREEN_HEIGHT+x]), 1.0f) * ((totalPixels - 1) / totalPixels) );
						else
							weighting = min(abs(focalDistances[y*SCREEN_HEIGHT+x]), 1.0f) * (1.0f / totalPixels);

						// Add contribution to final pixel colour
						finalColour += pixelColours[(y+z)*SCREEN_HEIGHT+(x+z2)] * weighting;
					}
				}
			}
			else
			{
				finalColour = pixelColours[y*SCREEN_HEIGHT+x];
			}

			PutPixelSDL( screen, x, y, finalColour );
		}
	}

	if( SDL_MUSTLOCK(screen) )
		SDL_UnlockSurface(screen);
		

	SDL_UpdateRect( screen, 0, 0, 0, 0 );

}
